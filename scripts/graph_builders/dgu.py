from scripts.graph_builders.gb_base import *

@register_pytree_node_class
class DGUGraphBuilder(GraphBuilder):
    def __init__(self, path):
        super().__init__(path, add_undirected_edges=False, add_self_loops=False)

    def _load_data(self, path):
        data = np.load(path, allow_pickle=True)
        config = data['config']
        state = data['state_trajectories']
        u = data['control_inputs']
        self._num_trajectories = state.shape[0]
        self._num_timesteps = state.shape[1]
        self._num_states = state.shape[2] + 1 # + 1 for ground node
        self._dt = config['dt']
        self.R = 1
        self.L = 1
        self.C = 1
        self.system_params = {
            'R': self.R,
            'L': self.L,
            'C': self.C,
        }
        self._control = jnp.array(u).squeeze()
        self._Q = jnp.array(state[:,:,0])
        self._Phi = jnp.array(state[:,:,1])
        self._V1 = jnp.zeros((self._num_trajectories, self._num_timesteps)) # round node
        self._V2 = jnp.array(state[:,:,2])
        self._V3 = jnp.array(state[:,:,3])
        self._V4 = jnp.array(state[:,:,4])
        self._jv = jnp.array(state[:,:,5])
        self._H = 0.5 * (self._Q**2 / self.C + self._Phi**2 / self.L)
        self._residuals = jnp.zeros((self._num_trajectories, self._num_timesteps))
        self.edge_idxs = np.array([3, 1, 2, 4, 0]) # V, R, L, I, C
        self.include_idxs = None
        self.node_idxs = None
        self.differential_vars = jnp.array([0, 1])
        self.algebraic_vars = jnp.array([2, 3, 4, 5, 6])

    def get_control(self, trajs, ts):
        return jnp.array(self._control[trajs, ts])
    
    def get_pred_data(self, graph):
        pred_Q = (graph.edges[4,0]).squeeze()
        pred_Phi = (graph.edges[2,0]).squeeze()
        pred_V1 = (graph.nodes[0]).squeeze()
        pred_V2 = (graph.nodes[1]).squeeze()
        pred_V3 = (graph.nodes[2]).squeeze()
        pred_V4 = (graph.nodes[3]).squeeze()
        pred_jv = (graph.edges[0,0]).squeeze()
        pred_H = (graph.globals[0]).squeeze()
        residual = jnp.array([jnp.sum((graph.globals[1:]))]).squeeze()
        return (pred_Q, pred_Phi, pred_V1, pred_V2, pred_V3, pred_V4, pred_jv, pred_H, residual) 
    
    def get_batch_pred_data(self, graphs) -> Sequence[jraph.GraphsTuple]:
        def f(carry, graph):
            return carry, self.get_pred_data(graph)
        
        _, batch_data = jax.lax.scan(f, None, graphs)
        
        return batch_data

    def get_exp_data(self, trajs, ts) -> Tuple:
        return (self._Q[trajs, ts], self._Phi[trajs, ts], self._V1[trajs, ts], self._V2[trajs, ts], self._V3[trajs, ts], self._V4[trajs, ts], self._jv[trajs, ts],self._H[trajs, ts], self._residuals[trajs, ts])
    
    def _get_norm_stats(self):
        # TODO
        norm_stats = ml_collections.ConfigDict()
        self._norm_stats = norm_stats
    
    def _setup_graph_params(self):
        self.n_node = jnp.array([4])
        self.n_edge = jnp.array([5])
        self.senders = jnp.array([0, 2, 3, 3, 0])
        self.receivers = jnp.array([1, 1, 2, 0, 3])

    def get_graph_from_state(self, state, control=None, system_params=None, set_nodes=False, set_ground_and_control=False, nodes=None, globals=None) -> jraph.GraphsTuple:
        Q = state[0]
        Phi = state[1]
        e = state[2:6]
        jv = state[6]
        I = control[0]
        V = control[1]
        nodes = state[2:6].reshape(-1,1)
        if set_ground_and_control:
            nodes = jnp.concatenate((jnp.array([[0]]), jnp.array([[V]]), nodes[2:]), axis=0)
        AR = jnp.array([[0, 1, -1, 0]]).T
        resistor_current = (AR.T @ e / 1.0).squeeze()
        # self.edge_idxs = np.array([3, 1, 2, 4, 0]) # V, R, L, I, C
        edges = jnp.array([[jv, 3], [jv, 1], [Phi, 2], [I, 4], [Q, 0]]) 
        n_node = jnp.array([4])
        n_edge = jnp.array([5])
        # TODO: how to decide senders and receivers (direction?)
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
        q = graph.edges[jnp.array([4]), 0]
        phi = graph.edges[jnp.array([2]), 0]
        e = graph.nodes.squeeze()
        jv = graph.edges[jnp.array([0]), 0]
        state = jnp.r_[q, phi, e, jv]
        return state
    
    def get_alg_vars_from_graph(self, graph):
        e = graph.nodes.squeeze()
        jv = graph.edges[0,0]
        y = jnp.r_[e, jv]
        return y
    
    def get_graph(self, traj_idx, t) -> jraph.GraphsTuple:
        nodes = jnp.array([[self._V1[traj_idx, t]], 
                           [self._V2[traj_idx, t]], 
                           [self._V3[traj_idx, t]], 
                           [self._V4[traj_idx, t]]])
        edges = jnp.array([[self._jv[traj_idx, t]],
                           [self._jv[traj_idx, t]],
                           [self._Phi[traj_idx, t]],
                           [self._control[traj_idx, t, 0]], # I
                           [self._Q[traj_idx, t]]]) 
        edges = jnp.concatenate((edges, self.edge_idxs.reshape(-1,1)), axis=1)
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
        return jnp.array([self._Q[traj_idx, t],
                          self._Phi[traj_idx, t],
                          self._V1[traj_idx, t],
                          self._V2[traj_idx, t],
                          self._V3[traj_idx, t],
                          self._V4[traj_idx, t],
                          self._jv[traj_idx, t]])
    
    def get_state_batch(self, traj_idxs, t0s) -> jnp.ndarray:
        def f(carry, idxs):
            return carry, self.get_state(*idxs)
        
        _, states = jax.lax.scan(f, None, (traj_idxs, t0s))
        return states
    
    def tree_flatten(self):
        children = ()
        aux_data = (self._dt, self.R, self.L, self.C, self.system_params, self._Q, self._Phi, self._V1, self._V2, self._V3, self._V4, self._jv, self._H, self._control, self._residuals, self._num_trajectories, self._num_timesteps, self._num_states, self.differential_vars, self.algebraic_vars)
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        obj = object.__new__(DGUGraphBuilder)
        obj._dt                   = aux_data[0]
        obj.R                     = aux_data[1]
        obj.L                     = aux_data[2]
        obj.C                     = aux_data[3]
        obj.system_params         = aux_data[4]
        obj._Q                    = aux_data[5]
        obj._Phi                  = aux_data[6]
        obj._V1                   = aux_data[7]
        obj._V2                   = aux_data[8]
        obj._V3                   = aux_data[9]
        obj._V4                   = aux_data[10]
        obj._jv                   = aux_data[11]
        obj._H                    = aux_data[12]
        obj._control              = aux_data[13]
        obj._residuals            = aux_data[14]
        obj._num_trajectories     = aux_data[15]
        obj._num_timesteps        = aux_data[16]
        obj._num_states           = aux_data[17]
        obj.differential_vars     = aux_data[18]
        obj.algebraic_vars        = aux_data[19]

        obj._setup_graph_params()
        return obj