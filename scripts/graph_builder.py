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
        N-Mass Spring Damper (DMSD) 
    """
    def __init__(self, path, add_undirected_edges, add_self_loops, mode, vel_history, control_history):
        super().__init__(path, add_undirected_edges, add_self_loops)
        self._mode = mode
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
    
    @jax.jit
    def get_graph(self, traj_idx, t) -> jraph.GraphsTuple:
        """ Need to make sure t > self._vel_history! """
        match self._mode:
            case 'acceleration':
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
            case 'position':
                raise NotImplementedError
            
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
        aux_data = (self._path, self._add_undirected_edges, self._add_self_loops, self._mode, self._vel_history, self._control_history, self._norm_stats, self._qs, self._dqs, self._vs, self._accs, self._control, self._m, self._k, self._b, self._dt, self._num_trajectories, self._num_timesteps) # static
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        obj = object.__new__(MSDGraphBuilder)
        obj._path                   = aux_data[0]
        obj._add_undirected_edges   = aux_data[1]
        obj._add_self_loops         = aux_data[2]
        obj._mode                   = aux_data[3]
        obj._vel_history            = aux_data[4]
        obj._control_history        = aux_data[5]
        obj._norm_stats             = aux_data[6]
        obj._qs                     = aux_data[7]
        obj._dqs                    = aux_data[8]
        obj._vs                     = aux_data[9]
        obj._accs                   = aux_data[10]
        obj._control                = aux_data[11]
        obj._m                      = aux_data[12]
        obj._k                      = aux_data[13]
        obj._b                      = aux_data[14]
        obj._dt                     = aux_data[15]
        obj._num_trajectories       = aux_data[16]
        obj._num_timesteps          = aux_data[17]

        obj._setup_graph_params()
        return obj
    
# @register_pytree_node_class
class PowerNetworkGraphBuilder(GraphBuilder):
    def __init__(self, path, add_undirected_edges, add_self_loops):
        super().__init__(path, add_undirected_edges, add_self_loops)

    def _load_data(self, path):
        return super()._load_data(path)
    
    def _get_norm_stats(self):
        return super()._get_norm_stats()
    
    def _setup_graph_params():
        pass
    
    def get_graph(self, **kwargs) -> jraph.GraphsTuple:
        return super().get_graph(**kwargs)
    
    def get_graph_batch(self, **kwargs) -> Sequence[jraph.GraphsTuple]:
        return super().get_graph_batch(**kwargs)
    
    def tree_flatten():
        pass
    
    @classmethod
    def tree_unflatten():
        pass
    
