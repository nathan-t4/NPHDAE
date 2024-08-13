from scripts.graph_builders.gb_base import *


@register_pytree_node_class
class LC1GraphBuilder(GraphBuilder):
    def __init__(self, path):
        super().__init__(path, add_undirected_edges=False, add_self_loops=False)

    def _load_data(self, path):
        data = np.load(path, allow_pickle=True)
        config = data['config']
        state = data['state_trajectories']
        u = data['control_inputs']
        self._num_trajectories = state.shape[0]
        self._num_timesteps = state.shape[1]
        self._num_states = 3
        self._dt = config['dt']
        self.C = jnp.array(config['C']).reshape(-1,1)
        self.C_prime = jnp.array(config['C_prime']).reshape(-1,1)
        self.L = jnp.array(config['L']).reshape(-1,1)
        self.system_params = {
            'C': self.C,
            'C_prime': self.C_prime,
            'L': self.L,
        }
        self._Q1 = jnp.array(state[:,:,0])
        self._Phi1 = jnp.array(state[:,:,1])
        self._Q3 = jnp.array(state[:,:,2])
        self._V1 = jnp.zeros((self._num_trajectories, self._num_timesteps))
        self._V2 = self._Q1 / self.C
        self._V3 = self._Q3 / self.C_prime
        self._H = 0.5 * (self._Q1**2 / self.C + self._Q3**2 / self.C_prime + self._Phi1**2 / self.L)
        self._residuals = jnp.zeros((self._num_trajectories, self._num_timesteps))
        self._control = jnp.array(u).squeeze()
        self.edge_idxs = np.array([0, 2, 0])
        self.node_idxs = None
        self.include_idxs = None
        self.differential_vars = jnp.array([0, 1, 2])
        self.algebraic_vars = jnp.array([3, 4, 5])
        self.J = jnp.array([[0, 1, 0],
                            [-1, 0, 1],
                            [0, -1, 0]])
        self.R = jnp.zeros((3,3))
        self.g = jnp.zeros((3,3))

    def get_control(self, trajs, ts):
        return self._control[trajs, ts]
    
    def get_pred_data(self, graph):
        pred_Q1 = (graph.edges[0,0]).squeeze()
        pred_Phi1 = (graph.edges[1,0]).squeeze()
        pred_Q3 = (graph.edges[2,0]).squeeze()
        pred_V1 = (graph.nodes[0]).squeeze()
        pred_V2 = (graph.nodes[1]).squeeze()
        pred_V3 = (graph.nodes[2]).squeeze()
        pred_H = (graph.globals[0]).squeeze()
        residual = jnp.array([jnp.sum((graph.globals[1:]))]).squeeze()
        return (pred_Q1, pred_Phi1, pred_Q3, pred_V1, pred_V2, pred_V3, pred_H, residual) 
    
    def get_batch_pred_data(self, graphs) -> Sequence[jraph.GraphsTuple]:
        def f(carry, graph):
            return carry, self.get_pred_data(graph)
        
        _, batch_data = jax.lax.scan(f, None, graphs)
        
        return batch_data

    def get_exp_data(self, trajs, ts) -> Tuple:
        return (self._Q1[trajs, ts], self._Phi1[trajs, ts], self._Q3[trajs, ts], self._V1[trajs, ts], self._V2[trajs, ts], self._V3[trajs, ts], self._H[trajs, ts], self._residuals[trajs, ts])
    
    def _get_norm_stats(self):
        norm_stats = ml_collections.ConfigDict()
        norm_stats.Q1 = ml_collections.ConfigDict({
            'mean': jnp.mean(self._Q1),
            'std': jnp.std(self._Q1),
        })
        norm_stats.Q3 = ml_collections.ConfigDict({
            'mean': jnp.mean(self._Q3),
            'std': jnp.std(self._Q3),
        })
        norm_stats.Phi1 = ml_collections.ConfigDict({
            'mean': jnp.mean(self._Phi1),
            'std': jnp.std(self._Phi1),
        })
        norm_stats.V2 = ml_collections.ConfigDict({
            'mean': jnp.mean(self._V2),
            'std': jnp.std(self._V2),
        })

        norm_stats.V3 = ml_collections.ConfigDict({
            'mean': jnp.mean(self._V3),
            'std': jnp.std(self._V3),
        })

        self._norm_stats = norm_stats
    
    def _setup_graph_params(self):
        self.n_node = jnp.array([3])
        self.n_edge = jnp.array([self._num_states])
        self.senders = jnp.array([0, 1, 2])
        self.receivers = jnp.array([1, 2, 0])

    def get_graph_from_state(self, state, control=None, system_params=None, set_nodes=False, set_ground_and_control=False, nodes=None, globals=None) -> jraph.GraphsTuple:
        Q1 = state[0]
        Q3 = state[1]
        Phi1 = state[2]
        nodes = state[3:6].reshape(-1,1)
        if set_nodes:
            V1 = 0
            V2 = Q1 / system_params['C']
            V3 = Q3 / system_params['C_prime']
            nodes = jnp.array([[V1], [V2], [V3]])
        if set_ground_and_control:
            nodes = jnp.concatenate((jnp.array([[0]]), nodes[1:]), axis=0)

        edges = jnp.array([[Q1, 0], [Phi1, 2], [Q3, 0]])
        n_node = jnp.array([3])
        n_edge = jnp.array([3])
        senders = self.senders
        receivers = self.receivers
        global_context = globals

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
    
    def get_state_from_graph(self, graph):
        q = graph.edges[jnp.array([0, 2]), 0]
        phi = graph.edges[jnp.array([1]), 0]
        e = graph.nodes.squeeze()
        state = jnp.r_[q, phi, e]
        return state
    
    def get_alg_vars_from_graph(self, graph):
        e = graph.nodes.squeeze()
        y = e
        return y
    
    def get_graph(self, traj_idx, t) -> jraph.GraphsTuple:
        nodes = jnp.array([[self._V1[traj_idx, t]], [self._V2[traj_idx, t]], [self._V3[traj_idx, t]]])
        edges = jnp.array([[self._Q1[traj_idx, t], 0], 
                           [self._Phi1[traj_idx, t], 2], 
                           [self._Q3[traj_idx, t], 0]])
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
    
    def get_state(self, traj_idx, t) -> jnp.ndarray:
        return jnp.array([self._Q1[traj_idx, t],
                          self._Q3[traj_idx, t],
                          self._Phi1[traj_idx, t],
                          self._V1[traj_idx, t],
                          self._V2[traj_idx, t],
                          self._V3[traj_idx, t]])
    
    def get_state_batch(self, traj_idxs, t0s) -> jnp.ndarray:
        def f(carry, idxs):
            return carry, self.get_state(*idxs)
        
        _, states = jax.lax.scan(f, None, (traj_idxs, t0s))
        return states
    
    def tree_flatten(self):
        children = ()
        aux_data = (self._dt, self.C, self.C_prime, self.L, self.system_params, self._Q1, self._Phi1, self._Q3, self._V1, self._V2, self._V3, self._H, self._control, self._residuals, self._num_trajectories, self._num_timesteps, self._num_states, self.differential_vars, self.algebraic_vars)
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        obj = object.__new__(LC1GraphBuilder)
        obj._dt                   = aux_data[0]
        obj.C                     = aux_data[1]
        obj.C_prime               = aux_data[2]
        obj.L                     = aux_data[3]
        obj.system_params         = aux_data[4]
        obj._Q1                   = aux_data[5]
        obj._Phi1                 = aux_data[6]
        obj._Q3                   = aux_data[7]
        obj._V1                   = aux_data[8]
        obj._V2                   = aux_data[9]
        obj._V3                   = aux_data[10]
        obj._H                    = aux_data[11]
        obj._control              = aux_data[12]
        obj._residuals            = aux_data[13]
        obj._num_trajectories     = aux_data[14]
        obj._num_timesteps        = aux_data[15]
        obj._num_states           = aux_data[16]
        obj.differential_vars     = aux_data[17]
        obj.algebraic_vars        = aux_data[18]

        obj._setup_graph_params()
        return obj