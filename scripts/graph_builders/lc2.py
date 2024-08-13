from scripts.graph_builders.gb_base import *


@register_pytree_node_class
class LC2GraphBuilder(GraphBuilder):
    def __init__(self, path):
        super().__init__(path, add_undirected_edges=False, add_self_loops=False)

    def _load_data(self, path):
        data = np.load(path, allow_pickle=True)
        config = data['config']
        state = data['state_trajectories']
        u = data['control_inputs']
        self._num_trajectories = state.shape[0]
        self._num_timesteps = state.shape[1]
        self._num_states = state.shape[2]
        self._dt = config['dt']
        self.C = jnp.array(config['C']).reshape(self._num_trajectories, 1)
        self.L = jnp.array(config['L']).reshape(self._num_trajectories, 1)
        self.system_params = {
            'C': self.C,
            'L': self.L,
        }

        self._Q = jnp.array(state[:,:,0]).reshape(self._num_trajectories, self._num_timesteps)
        self._Phi = jnp.array(state[:,:,1]).reshape(self._num_trajectories, self._num_timesteps)

        self._control = jnp.array(u)

        self._Vc = self._Q / self.C
        self._H = 0.5 * (self._Q**2 / self.C + self._Phi**2 / self.L)

        self.J = jnp.array([[0, 1],
                            [-1, 0]])
        self.R = jnp.zeros((2,2))
        self.g = jnp.zeros((2,2))

    def get_control(self, trajs, ts):
        return self._control[trajs, ts]
    
    def get_pred_data(self, graph):
        pred_Q = (graph.edges[0,0]).squeeze()
        pred_Phi = (graph.edges[1,0]).squeeze()
        pred_H = (graph.globals).squeeze()
        return (pred_Q, pred_Phi, pred_H)
    
    def get_exp_data(self, trajs, ts):
        return (self._Q[trajs, ts], self._Phi[trajs, ts], self._H[trajs, ts])
    
    def _get_norm_stats(self):
        norm_stats = ml_collections.ConfigDict()
        norm_stats.Q = ml_collections.ConfigDict({
            'mean': jnp.mean(self._Q),
            'std': jnp.std(self._Q),
        })
        
        norm_stats.Phi = ml_collections.ConfigDict({
            'mean': jnp.mean(self._Phi),
            'std': jnp.std(self._Phi),
        })

        norm_stats.Vc = ml_collections.ConfigDict({
            'mean': jnp.mean(self._Vc),
            'std': jnp.std(self._Vc),
        })

        self._norm_stats = norm_stats
    
    def _setup_graph_params(self):
        self.n_node = jnp.array([3])
        self.n_edge = jnp.array([self._num_states])
        self.senders = jnp.array([2, 1])
        self.receivers = jnp.array([1, 0])

    def get_graph_from_state(self, state, control, system_params=None, set_nodes=False, set_ground_and_control=False, nodes=None, globals=None) -> jraph.GraphsTuple:
        Q = state[0]
        Phi = state[1]
        _, volt = control
        nodes = nodes
        if set_nodes:
            Vc = Q / system_params['C']
            I = Phi / system_params['L']
            nodes = jnp.array([[volt], [Vc], [0]])
        elif set_ground_and_control:
            nodes = jnp.concatenate((jnp.array([volt]), nodes[1], jnp.array([0])), axis=0).reshape(-1,1)

        edges = jnp.array([[Q, 0], [Phi, 1]])
        global_context = globals
        n_node = jnp.array([3])
        n_edge = jnp.array([2])
        senders = jnp.array([2, 1])
        receivers = jnp.array([1, 0])

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
        nodes = jnp.array([[self._control[traj_idx, t, 1]], [self._Vc[traj_idx, t]], [0]])
        edges = jnp.array([[self._Q[traj_idx, t], 0],
                           [self._Phi[traj_idx, t], 1]])
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
        aux_data = (self._dt, self.C, self.L, self.system_params, self._Q, self._Phi, self._control, self._Vc, self._H, self._num_trajectories, self._num_timesteps, self._num_states, self.J, self.R, self.g)
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        obj = object.__new__(LC2GraphBuilder)
        obj._dt                   = aux_data[0]
        obj.C                     = aux_data[1]
        obj.L                     = aux_data[2]
        obj.system_params         = aux_data[3]
        obj._Q                    = aux_data[4]
        obj._Phi                  = aux_data[5]
        obj._control              = aux_data[6]
        obj._Vc                   = aux_data[7]
        obj._H                    = aux_data[8]
        obj._num_trajectories     = aux_data[9]
        obj._num_timesteps        = aux_data[10]
        obj._num_states           = aux_data[11]
        obj.J                     = aux_data[12]
        obj.R                     = aux_data[13]
        obj.g                     = aux_data[14]
        obj._setup_graph_params()
        return obj