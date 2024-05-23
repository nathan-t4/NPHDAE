import jraph
import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
from jax.tree_util import register_pytree_node_class
from typing import Sequence
from utils.graph_utils import add_edges

class GraphBuilder():
    def __init__(self, path, add_undirected_edges, add_self_loops):
        self._path = path
        self._add_undirected_edges = add_undirected_edges
        self._add_self_loops = add_self_loops
        self._load_data(self._path)
        self._get_norm_stats()
        self._setup_graph_params()

    def init(**kwargs):
        raise NotImplementedError

    def _load_data(self, path):
        raise NotImplementedError

    def _get_norm_stats(self):
        raise NotImplementedError
    
    def _setup_graph_params():
        raise NotImplementedError
    
    def get_graph(self, **kwargs) -> jraph.GraphsTuple:
        raise NotImplementedError
    
    def get_graph_batch(self, **kwargs) -> Sequence[jraph.GraphsTuple]:
        raise NotImplementedError
    
    def tree_flatten():
        raise NotImplementedError
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        raise NotImplementedError
    
@register_pytree_node_class
class MSDGraphBuilder(GraphBuilder):
    """ 
        N-Mass Spring Damper (DMSD) from dataset
    """
    def __init__(self, path, add_undirected_edges, add_self_loops, vel_history, control_history):
        super().__init__(path, add_undirected_edges, add_self_loops)
        self._vel_history = vel_history
        self._control_history = control_history
    
    def _load_data(self, path):
        """
            The resulting dataset has dimensions [num_trajectories, num_timesteps, (qs, dqs, ps, accs)]

        """
        data = np.load(path, allow_pickle=True)
        state = data['state_trajectories']
        config = data['config']
        control = data['control_inputs']
        # Number of trajectories in dataset
        self._num_trajectories = state.shape[0]
        # Number of timesteps per trajectory
        self._num_timesteps = state.shape[1]
        self._dt = config['dt']
        # Control
        # self._control = jnp.concatenate((jnp.zeros(control.shape), control), axis=-1)
        self._control = jnp.array(control)
        # Masses
        # self._m = jnp.array([config['m1'], config['m2']]).T
        self._m = jnp.array(config['m'])
        # Spring constants
        # self._k = jnp.array([config['k1'], config['k2']]).T
        self._k = jnp.array(config['k'])
        # Damper constants
        # self._b = jnp.array([config['b1'], config['b2']]).T
        self._b = jnp.array(config['b'])
        # Absolute position
        self._qs = jnp.array(state[:,:,::2])
        # Relative positions
        # self._dqs = jnp.array([]) # TODO: for when N = 1
        self._dqs = jnp.diff(self._qs, axis=-1)
        # Conjugate momenta
        # self._ps = jnp.array(state[:,:,1::2])
        # Velocities
        self._vs = jnp.array(state[:,:,1::2]) / jnp.expand_dims(self._m, 1)  # reshape m to fit shape of velocity
        # Accelerations
        self._accs = jnp.diff(self._vs, axis=1) / self._dt
        final_acc = jnp.expand_dims(self._accs[:,-1,:], axis=1) # duplicate final acceleration
        self._accs = jnp.concatenate((self._accs, final_acc), axis=1) # add copy of final acceleration to end of accs
        # data = jnp.concatenate((self._qs, self._dqs, self._ps, self._accs), axis=-1)
        # data = jax.lax.stop_gradient(data)
        # self._data = data
    
    def _get_norm_stats(self):
        norm_stats = ml_collections.ConfigDict()
        norm_stats.position = ml_collections.ConfigDict({
            'mean': jnp.mean(self._qs),
            'std': jnp.std(self._qs),
        })
        norm_stats.velocity = ml_collections.ConfigDict({
            'mean': jnp.mean(self._vs),
            'std': jnp.std(self._vs),
        })
        norm_stats.acceleration = ml_collections.ConfigDict({
            'mean': jnp.mean(self._accs),
            'std': jnp.std(self._accs),
        })
        norm_stats.control = ml_collections.ConfigDict({
            'mean': jnp.mean(self._control),
            'std': jnp.std(self._control)
        })

        self._norm_stats = norm_stats
    
    def _setup_graph_params(self):
        self.n_node = jnp.array([jnp.shape(self._qs)[-1]])
        self.n_edge = jnp.array([jnp.shape(self._dqs)[-1]])
        self.senders = jnp.arange(0, jnp.shape(self._qs)[-1]-1)
        self.receivers = jnp.arange(1, jnp.shape(self._qs)[-1])

    def get_graph_from_state(self, x) -> jraph.GraphsTuple:
        q = x[::2]
        qdot = x[1::2]
        nodes = jnp.column_stack((q, qdot))
    
    @jax.jit
    def get_graph(self, traj_idx, t) -> jraph.GraphsTuple:
        """ Need to make sure t > self._vel_history! """
        vs_history = []                
        [vs_history.append(self._vs[traj_idx, t-k]) for k in reversed(range(self._vel_history))]
        vs_history = jnp.asarray(vs_history).T

        control_history = []
        [control_history.append(self._control[traj_idx, t-k, 1::2]) for k in reversed(range(self._control_history))]
        control_history = jnp.asarray(control_history).T
        # Node features are current position, velocity history, current velocity
        nodes = jnp.column_stack((self._qs[traj_idx, t], vs_history, control_history))
        # Edge features are relative positions
        edges = self._dqs[traj_idx, t].reshape((-1,1))
        # Global features are time, q0, v0, a0
        # global_context = jnp.concatenate((jnp.array([t]), self._qs[traj_idx, 0], self._vs[traj_idx, 0], self._accs[traj_idx, 0])).reshape(-1,1)
    
        # Global features are None
        global_context = None

        graph =  jraph.GraphsTuple(
                    nodes=nodes,
                    edges=edges,
                    senders=self.senders,
                    receivers=self.receivers,
                    n_node=self.n_node,
                    n_edge=self.n_edge,
                    globals=global_context)
        
        graph = add_edges(graph, self._add_undirected_edges, self._add_self_loops)

        return graph
    
    @jax.jit
    def get_graph_batch(self, traj_idxs, t0s) -> Sequence[jraph.GraphsTuple]:
        def f(carry, idxs):
            return carry, self.get_graph(*idxs)
        
        _, graphs = jax.lax.scan(f, None, (traj_idxs, t0s))
        
        return graphs

    def tree_flatten(self):
        children = () # dynamic
        aux_data = (self._path, self._add_undirected_edges, self._add_self_loops, self._vel_history, self._control_history, self._norm_stats, self._qs, self._dqs, self._vs, self._accs, self._control, self._m, self._k, self._b, self._dt, self._num_trajectories, self._num_timesteps) # static
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        obj = object.__new__(MSDGraphBuilder)
        obj._path                   = aux_data[0]
        obj._add_undirected_edges   = aux_data[1]
        obj._add_self_loops         = aux_data[2]
        obj._vel_history            = aux_data[3]
        obj._control_history        = aux_data[4]
        obj._norm_stats             = aux_data[5]
        obj._qs                     = aux_data[6]
        obj._dqs                    = aux_data[7]
        obj._vs                     = aux_data[8]
        obj._accs                   = aux_data[9]
        obj._control                = aux_data[10]
        obj._m                      = aux_data[11]
        obj._k                      = aux_data[12]
        obj._b                      = aux_data[13]
        obj._dt                     = aux_data[14]
        obj._num_trajectories       = aux_data[15]
        obj._num_timesteps          = aux_data[16]

        obj._setup_graph_params()
        return obj
    
@register_pytree_node_class
class LCGraphBuilder(GraphBuilder):
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
        self.C = config['C'][0]
        self.L = config['L'][0]

        self._Q = jnp.array(state[:,:,0])
        self._Phi = jnp.array(state[:,:,1])
        self._V = self._Q / self.C

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
        V = Q / self.C
        nodes = jnp.array([[0], [V]])
        edges = jnp.array([[Q], [Phi]])
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
    
    @jax.jit
    def get_graph(self, traj_idx, t) -> jraph.GraphsTuple:
        nodes = jnp.array([[0], [self._V[traj_idx, t]]])
        edges = jnp.array([[self._Q[traj_idx, t]], [self._Phi[traj_idx, t]]])
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

    @jax.jit
    def get_graph_batch(self, traj_idxs, t0s) -> Sequence[jraph.GraphsTuple]:
        def f(carry, idxs):
            return carry, self.get_graph(*idxs)
        
        _, graphs = jax.lax.scan(f, None, (traj_idxs, t0s))
        
        return graphs
    
    def tree_flatten(self):
        children = ()
        aux_data = (self._dt, self.C, self.L, self._Q, self._Phi, self._V, self._num_trajectories, self._num_timesteps, self._num_states)
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        obj = object.__new__(LCGraphBuilder)
        obj._dt                   = aux_data[0]
        obj.C                     = aux_data[1]
        obj.L                     = aux_data[2]
        obj._Q                    = aux_data[3]
        obj._Phi                  = aux_data[4]
        obj._V                    = aux_data[5]
        obj._num_trajectories     = aux_data[6]
        obj._num_timesteps        = aux_data[7]
        obj._num_states           = aux_data[8]

        obj._setup_graph_params()
        return obj

@register_pytree_node_class
class CoupledLCGraphBuilder(GraphBuilder):
    def __init__(self, path, C, C_prime, L):
        super().__init__(path, add_undirected_edges=False, add_self_loops=False)
        self.C = C
        self.C_prime = C_prime
        self.L = L

    def _load_data(self, path):
        data = np.load(path, allow_pickle=True)
        config = data['config']
        state = data['state_trajectories']
        self._num_trajectories = state.shape[0]
        self._num_timesteps = state.shape[1]
        self._num_states = state.shape[2]
        self._dt = config['dt']

        self._Q1 = state[:,:,0]
        self._Phi1 = state[:,:,1]
        self._V2 = self._Q1 / self.C

        self._Q2 = state[:,:,2]
        self._Phi2 = state[:,:,3]
        self._V4 = self._Q2 / self.C

        self._Q3 = -(self._Q1 + self._Q2)
        self._V3 = self._Q3 / self.C_prime

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

        norm_stats.V2 = ml_collections.ConfigDict({
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
    
    @jax.jit
    def get_graph(self, traj_idx, t) -> jraph.GraphsTuple:
        ground_voltage = jnp.zeros(jnp.shape(self._V2[traj_idx, t]))
        nodes = jnp.concatenate((ground_voltage, self._V2[traj_idx, t], self._V3[traj_idx, t], self._V4[traj_idx, t]))
        edges = jnp.concatenate((self._Q1[traj_idx, t], self._Phi1[traj_idx, t] / self.L, self._Q3[traj_idx, t], self._Phi2[traj_idx, t] / self.L, self._Q2[traj_idx, t])) # TODO: remove (self.L)**(-1)?
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

    @jax.jit
    def get_graph_batch(self, traj_idxs, t0s) -> Sequence[jraph.GraphsTuple]:
        def f(carry, idxs):
            return carry, self.get_graph(*idxs)
        
        _, graphs = jax.lax.scan(f, None, (traj_idxs, t0s))
        
        return graphs
    
    def tree_flatten(self):
        children = ()
        aux_data = (self._dt, self.C, self.C_prime, self.L, self._num_trajectories, self._num_timesteps, self._num_states, self._Q1, self._Phi1, self._V1, self._Q2, self._Phi2, self._V2, self._Q3, self._V3)
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        obj = object.__new__(LCGraphBuilder)
        obj._dt                   = aux_data[0]
        obj.C                     = aux_data[1]
        obj.C_prime               = aux_data[2]
        obj.L                     = aux_data[3]
        obj._num_trajectories     = aux_data[4]
        obj._num_timesteps        = aux_data[5]
        obj._num_states           = aux_data[6]
        obj._Q1                   = aux_data[7]
        obj._Phi1                 = aux_data[8]
        obj._V1                   = aux_data[9]
        obj._Q2                   = aux_data[10]
        obj._Phi2                 = aux_data[11]
        obj._V2                   = aux_data[12]
        obj._Q3                   = aux_data[13]
        obj._V3                   = aux_data[14]

        obj._setup_graph_params()
        return obj