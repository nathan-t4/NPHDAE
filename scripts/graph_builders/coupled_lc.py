from scripts.graph_builders.gb_base import *

@register_pytree_node_class
class CoupledLCGraphBuilder(GraphBuilder):
    def __init__(self, path):
        super().__init__(path, add_undirected_edges=False, add_self_loops=False)

    def _load_data(self, path):
        data = np.load(path, allow_pickle=True)
        config = data['config']
        state = data['state_trajectories']
        self._num_trajectories = state.shape[0]
        self._num_timesteps = state.shape[1]
        self._num_states = 5
        self._dt = config['dt']
        self.system_params = {
            'C': jnp.array(config['C']).reshape(-1,1),
            'C_prime': jnp.array(config['C_prime']).reshape(-1,1),
            'L': jnp.array(config['L']).reshape(-1,1),
        }
        self._control = jnp.array(data['control_inputs'])

        self._Q1 = jnp.array(state[:,:,0])
        self._Phi1 = jnp.array(state[:,:,1])
        self._V2 = self._Q1 / self.system_params['C']

        self._Q3 = jnp.array(state[:,:,2])
        self._V3 = self._Q3 / self.system_params['C_prime']

        self._Q2 = jnp.array(state[:,:,3])
        self._Phi2 = jnp.array(state[:,:,4])
        self._V4 = self._Q2 / self.system_params['C']

        self._H = 0.5 * (self._Q1 ** 2 / self.system_params['C'] + self._Q2 ** 2 / self.system_params['C'] + self._Q3 ** 2 / self.system_params['C_prime'] + self._Phi1 ** 2 / self.system_params['L'] + self._Phi2 ** 2 / self.system_params['L'])

        self.edge_idxs = np.array([[0,2,3]])
        self.node_idxs = None
        self.J = jnp.array([[0, 1, 0, 0, 0],
                            [-1, 0, 1, 0, 0],
                            [0, -1, 0, 0, -1],
                            [0, 0, 0, 0, 1],
                            [0, 0, 1, -1, 0]])
        self.R = jnp.zeros((5,5))
        self.g = jnp.zeros((5,5))

    def get_control(self, trajs, ts):
        return self._control[trajs, ts]
    
    def get_pred_data(self, graph):
        pred_Q1 = (graph.edges[0,0]).squeeze()
        pred_Phi1 = (graph.edges[1,0]).squeeze()
        pred_Q3 = (graph.edges[2,0]).squeeze()
        pred_Q2 = (graph.edges[3,0]).squeeze()
        pred_Phi2 = (graph.edges[4,0]).squeeze()
        pred_H = (graph.globals).squeeze()
        return (pred_Q1, pred_Phi1, pred_Q3, pred_Q2, pred_Phi2, pred_H)
    
    def get_exp_data(self, trajs, ts):
        return (self._Q1[trajs, ts], self._Phi1[trajs, ts], self._Q3[trajs, ts], self._Q2[trajs, ts], self._Phi2[trajs, ts], self._H[trajs, ts])
    
    def _get_norm_stats(self):
        norm_stats = ml_collections.ConfigDict()
        norm_stats.Q1 = ml_collections.ConfigDict({
            'mean': jnp.mean(self._Q1),
            'std': jnp.std(self._Q1),
        })
        
        norm_stats.Phi1 = ml_collections.ConfigDict({
            'mean': jnp.mean(self._Phi1),
            'std': jnp.std(self._Phi1),
        })

        norm_stats.V2 = ml_collections.ConfigDict({
            'mean': jnp.mean(self._V2),
            'std': jnp.std(self._V2),
        })

        norm_stats.Q2 = ml_collections.ConfigDict({
            'mean': jnp.mean(self._Q2),
            'std': jnp.std(self._Q2),
        })
        
        norm_stats.Phi2 = ml_collections.ConfigDict({
            'mean': jnp.mean(self._Phi2),
            'std': jnp.std(self._Phi2),
        })

        norm_stats.V4 = ml_collections.ConfigDict({
            'mean': jnp.mean(self._V4),
            'std': jnp.std(self._V4),
        })

        norm_stats.Q3 = ml_collections.ConfigDict({
            'mean': jnp.mean(self._Q3),
            'std': jnp.std(self._Q3),
        })

        norm_stats.V3 = ml_collections.ConfigDict({
            'mean': jnp.mean(self._V3),
            'std': jnp.std(self._V3),
        })
        
        self._norm_stats = norm_stats
    
    def _setup_graph_params(self):
        self.n_node = jnp.array([4])
        self.n_edge = jnp.array([5])
        self.senders = jnp.array([0, 1, 0, 3, 0])
        self.receivers = jnp.array([1, 2, 2, 2, 3])

    def get_graph_from_state(self, state, control, system_params=None, set_nodes=False, set_ground_and_control=False, nodes=None, globals=None):
        Q1, Phi1, Q3, Q2, Phi2 = state
        nodes = nodes
        if set_nodes:
            V2 = Q1 / system_params['C']
            V3 = Q3 / system_params['C_prime']
            V4 = Q2 / system_params['C']
            nodes = jnp.array([[0], [V2], [V3], [V4]])
        elif set_ground_and_control:
            nodes = jnp.concatenate((jnp.array([[0]]), nodes[1:]), axis=0).reshape(-1,1)

        edges = jnp.array([[Q1, 0], [Phi1, 1], [Q3, 0], [Q2, 0], [Phi2, 1]])
        global_context = globals
        n_node = jnp.array([4])
        n_edge = jnp.array([5])
        senders = jnp.array([0, 1, 0, 3, 0])
        receivers = jnp.array([1, 2, 2, 2, 3])

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
        nodes = jnp.array([[0], [self._V2[traj_idx, t]], [self._V3[traj_idx, t]], [self._V4[traj_idx, t]]])
        edges = jnp.array([[self._Q1[traj_idx, t], 0],
                           [self._Phi1[traj_idx, t], 1],
                           [self._Q3[traj_idx, t], 0],
                           [self._Q2[traj_idx, t], 0],
                           [self._Phi2[traj_idx, t], 1]])
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
        aux_data = (self._dt, self.system_params, self._num_trajectories, self._num_timesteps, self._num_states, self._Q1, self._Phi1, self._V2, self._Q2, self._Phi2, self._V4, self._Q3, self._V3, self._H, self._control, self._norm_stats, self.J, self.R, self.g)
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        obj = object.__new__(CoupledLCGraphBuilder)
        obj._dt                   = aux_data[0]
        obj.system_params         = aux_data[1]
        obj._num_trajectories     = aux_data[2]
        obj._num_timesteps        = aux_data[3]
        obj._num_states           = aux_data[4]
        obj._Q1                   = aux_data[5]
        obj._Phi1                 = aux_data[6]
        obj._V2                   = aux_data[7]
        obj._Q2                   = aux_data[8]
        obj._Phi2                 = aux_data[9]
        obj._V4                   = aux_data[10]
        obj._Q3                   = aux_data[11]
        obj._V3                   = aux_data[12]
        obj._H                    = aux_data[13]
        obj._control              = aux_data[14]
        obj._norm_stats           = aux_data[15]
        obj.J                     = aux_data[16]
        obj.R                     = aux_data[17]
        obj.g                     = aux_data[18]
        obj._setup_graph_params()
        return obj