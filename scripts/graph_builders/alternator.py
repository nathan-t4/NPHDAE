from scripts.graph_builders.gb_base import *

@register_pytree_node_class
class AlternatorGraphBuilder(GraphBuilder):
    def __init__(self, path):
        super().__init__(path, add_undirected_edges=False, add_self_loops=False)

    def _load_data(self, path):
        data = np.load(path, allow_pickle=True)
        config = data['config']
        state = data['state_trajectories']
        self._num_trajectories = state.shape[0]
        self._num_timesteps = state.shape[1]
        self._num_states = state.shape[2]
        self._dt = config['dt']

        params = ('rm', 'rr', 'd', 'M', 'L_aa0', 'L_ab0', 'L_afd', 'L_akd', 'L_akq', 'L_ffd', 'L_kkd', 'L_kkq')

        self.system_params = {f'{p}': jnp.array(config[f'{p}']).reshape(-1,1).reshape(-1,1) for p in params}

        self._inductances = jnp.concatenate((
            self.system_params['L_aa0'],
            self.system_params['L_ab0'],
            self.system_params['L_afd'],
            self.system_params['L_akd'],
            self.system_params['L_akq'],
            self.system_params['L_ffd'],
            self.system_params['L_kkd'],
            self.system_params['L_kkq'],
        )).squeeze()

        self.edge_idxs = np.array([[0,1,2,3,4,5], [6,None,None,None,None,None]])
        self.node_idxs = None

        self.J = jnp.array([[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, -1],
                            [0, 0, 0, 0, 0, 0, 1, 0]])

        rm = self.system_params['rm'].squeeze()[0]
        rr = self.system_params['rr'].squeeze()[0]
        d = self.system_params['d'].squeeze()[0]
        Rsl = jnp.diag(jnp.array([rm, rm, rm]))
        Rr = jnp.diag(jnp.array([rr, rr, rr]))
        self.R = jax.scipy.linalg.block_diag(Rsl, Rr, d, 0)

        self.g = jnp.array([[0, 0],
                            [0, 0],
                            [0, 0],
                            [1, 0],
                            [0, 0],
                            [0, 0],
                            [0, 1],
                            [0, 0]])

        self._control = jnp.array(data['control_inputs'])

        self._PhiS = jnp.array(state[:,:,0:3])
        self._PhiR = jnp.array(state[:,:,3:6])
        self._p = jnp.array(state[:,:,6])
        self._theta = jnp.array(state[:,:,7])

        def get_L(_, t, inductances):
            L_aa0, L_ab0, L_afd, L_akd, L_akq, L_ffd, L_kkd, L_kkq = inductances
            L_ess = jnp.array([[L_aa0, -L_ab0, -L_ab0],
                            [-L_ab0, L_aa0, -L_ab0],
                            [-L_ab0, -L_ab0, L_aa0]])
    
            L_ers = jnp.array([
                [L_afd * jnp.cos(t), L_akd * jnp.cos(t), -L_akq * jnp.sin(t)],
                [L_afd * jnp.cos(t - 2 * jnp.pi / 3), L_akd * jnp.cos(t - 2 * jnp.pi / 3), -L_akq * jnp.sin(t - 2 * jnp.pi / 3)],
                [L_afd * jnp.cos(t + 2 * jnp.pi / 3), L_akd * jnp.cos(t + 2 * jnp.pi / 3), -L_akq * jnp.sin(t + 2 * jnp.pi / 3)]
            ])

            L_err = jnp.array([[L_ffd, L_akd, 0],
                               [L_akd, L_kkd, 0],
                               [0, 0, L_kkq]])
            
            L = jnp.block([[L_ess, L_ers], # Inductance matrix
                           [L_ers.T, L_err]])
            
            return _, L

        vmap_get_L = jax.vmap(partial(get_L, inductances=self._inductances), in_axes=(None,0))

        _, self._Ls = jax.lax.scan(vmap_get_L, None, self._theta)

        Phi = jnp.concatenate((self._PhiS, self._PhiR), axis=-1)
        Phi = jnp.expand_dims(Phi, -1)

        PE = jax.lax.batch_matmul(Phi.transpose((0,1,3,2)), 
                                  jax.lax.batch_matmul(jnp.linalg.inv(self._Ls), Phi))
        PE = PE.squeeze()

        self._H = 0.5 * PE + 0.5 * (self._p ** 2 / self.system_params['M'])

    def get_control(self, trajs, ts):
        return self._control[trajs, ts]
    
    def get_pred_data(self, graph):
        pred_PhiSa, pred_PhiSb, pred_PhiSc = graph.edges[0,0].squeeze(), graph.edges[1,0].squeeze(), graph.edges[2,0].squeeze()
        pred_PhiRf, pred_PhiRkd, pred_PhiRkq = graph.edges[3,0].squeeze(), graph.edges[4,0].squeeze(), graph.edges[5,0].squeeze()
        pred_p = (graph.edges[6,0]).squeeze()
        pred_theta = (graph.edges[7,0]).squeeze()
        pred_H = (graph.globals).squeeze()
        return (pred_PhiSa, pred_PhiSb, pred_PhiSc, pred_PhiRf, pred_PhiRkd, pred_PhiRkq, pred_p, pred_theta, pred_H)
    
    def get_exp_data(self, trajs, ts):
        PhiSa, PhiSb, PhiSc = self._PhiS[trajs, ts, 0], self._PhiS[trajs, ts, 1], self._PhiS[trajs, ts, 2]
        PhiRf, PhiRkd, PhiRkq = self._PhiR[trajs, ts, 0], self._PhiR[trajs, ts, 1], self._PhiR[trajs, ts, 2]
        return (PhiSa, PhiSb, PhiSc, PhiRf, PhiRkd, PhiRkq, self._p[trajs, ts], self._theta[trajs, ts], self._H[trajs, ts])
    
    def _get_norm_stats(self):
        self._norm_stats = None
    
    def _setup_graph_params(self):
        self.n_node = jnp.array([3])
        self.n_edge = jnp.array([8])
        self.senders = jnp.array([1, 1, 1, 0, 0, 0, 0, 0])
        self.receivers = jnp.array([2, 2, 2, 1, 1, 1, 1, 1])

    def get_graph_from_state(self, state, control, system_params=None, set_nodes=False, set_ground_and_control=False, nodes=None, globals=None):
        PhiSa, PhiSb, PhiSc = state[:3]
        PhiRf, PhiRkd, PhiRkq = state[3:6]
        p = state[6]
        theta = state[7]
        nodes = nodes
        if set_ground_and_control:
            nodes = jnp.concatenate((jnp.array([[0.0]]), nodes[1:]), axis=0).reshape(-1,1)

        edges = jnp.array([[PhiSa, 0], [PhiSb, 0], [PhiSc, 0], [PhiRf, 0], [PhiRkd, 0], [PhiRkq, 0],  [p, 1], [theta, 2]])
        global_context = globals
        n_node = jnp.array([3])
        n_edge = jnp.array([8])
        senders = jnp.array([0, 0, 0, 1, 1, 1, 0, 0])
        receivers = jnp.array([1, 1, 1, 2, 2, 2, 1, 1,])

        graph = jraph.GraphsTuple(
            nodes=nodes,
            edges=edges,
            globals=global_context,
            n_node=n_node,
            n_edge=n_edge,
            senders=senders,
            receivers=receivers,
        )

        return graph
    
    def get_graph(self, traj_idx, t) -> jraph.GraphsTuple:
        nodes = jnp.array([[0.0], [0.0], [0.0]])
        edges = jnp.array([[self._PhiS[traj_idx, t, 0], 0],
                           [self._PhiS[traj_idx, t, 1], 0],
                           [self._PhiS[traj_idx, t, 2], 0],
                           [self._PhiR[traj_idx, t, 0], 0],
                           [self._PhiR[traj_idx, t, 1], 0],
                           [self._PhiR[traj_idx, t, 2], 0],
                           [self._p[traj_idx, t], 1],
                           [self._theta[traj_idx, t], 2],])
        global_context = None

        graph =  jraph.GraphsTuple(
                    nodes=nodes,
                    edges=edges,
                    senders=self.senders,
                    receivers=self.receivers,
                    n_node=self.n_node,
                    n_edge=self.n_edge,
                    globals=global_context)
        
        return graph

    def get_graph_batch(self, traj_idxs, t0s) -> Sequence[jraph.GraphsTuple]:
        def f(carry, idxs):
            return carry, self.get_graph(*idxs)
        
        _, graphs = jax.lax.scan(f, None, (traj_idxs, t0s))
        
        return graphs
    
    def tree_flatten(self):
        children = ()
        aux_data = (self._dt, self.system_params, self._num_trajectories, self._num_timesteps, self._num_states, self._PhiS, self._PhiR, self._p, self._theta, self._H, self._control, self._norm_stats, self.J, self.R, self.g)
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        obj = object.__new__(AlternatorGraphBuilder)
        obj._dt                   = aux_data[0]
        obj.system_params         = aux_data[1]
        obj._num_trajectories     = aux_data[2]
        obj._num_timesteps        = aux_data[3]
        obj._num_states           = aux_data[4]
        obj._PhiS                 = aux_data[5]
        obj._PhiR                 = aux_data[6]
        obj._p                    = aux_data[7]
        obj._theta                = aux_data[8]
        obj._H                    = aux_data[9]
        obj._control              = aux_data[10]
        obj._norm_stats           = aux_data[11]
        obj.J                     = aux_data[12]
        obj.R                     = aux_data[13]
        obj.g                     = aux_data[14]
        obj._setup_graph_params()
        return obj
    