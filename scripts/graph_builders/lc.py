from scripts.graph_builders.gb_base import *

@register_pytree_node_class
class LCGraphBuilder(GraphBuilder):
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
        self.system_params = {
            'C': config['C'][0],
            'L': config['L'][0],
        }

        self._Q = jnp.array(state[:,:,0])
        self._Phi = jnp.array(state[:,:,1])
        self._V = self._Q / self.system_params['C']
        self._H = 0.5 * (self._Q**2 / self.system_params['C'] + self._Phi**2 / self.system_params['L'])
        self._control = jnp.array(u).squeeze()
    
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

        norm_stats.V = ml_collections.ConfigDict({
            'mean': jnp.mean(self._V),
            'std': jnp.std(self._V),
        })

        self._norm_stats = norm_stats
    
    def _setup_graph_params(self):
        self.n_node = jnp.array([2])
        self.n_edge = jnp.array([2])
        self.senders = jnp.array([0, 1])
        self.receivers = jnp.array([1, 0])

    def get_graph_from_state(self, state) -> jraph.GraphsTuple:
        Q = state[0]
        Phi = state[1]
        V = Q / self.system_params['C']
        nodes = jnp.array([[0], [V]])
        edges = jnp.array([[Q, 0], [Phi, 1]])
        global_context = None

        graph = jraph.GraphsTuple(
            nodes=nodes,
            edges=edges,
            globals=global_context,
            n_node=self.n_node,
            n_edge=self.n_edge,
            senders=self.senders,
            receivers=self.receivers,
        )

        return graph
    
    def get_graph(self, traj_idx, t) -> jraph.GraphsTuple:
        nodes = jnp.array([[0], [self._V[traj_idx, t]]])
        edges = jnp.array([[self._Q[traj_idx, t], 0], [self._Phi[traj_idx, t], 1]])
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
        aux_data = (self._dt, self.system_params, self._Q, self._Phi, self._V, self._H, self._num_trajectories, self._num_timesteps, self._num_states)
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        obj = object.__new__(LCGraphBuilder)
        obj._dt                   = aux_data[0]
        obj.system_params         = aux_data[1]
        obj._Q                    = aux_data[2]
        obj._Phi                  = aux_data[3]
        obj._V                    = aux_data[4]
        obj._H                    = aux_data[5]
        obj._num_trajectories     = aux_data[6]
        obj._num_timesteps        = aux_data[7]
        obj._num_states           = aux_data[8]

        obj._setup_graph_params()
        return obj
    