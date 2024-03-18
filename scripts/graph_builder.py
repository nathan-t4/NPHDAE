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
class DMSDGraphBuilder(GraphBuilder):
    """ 
        Double Mass Spring Damper (DMSD) 
    """
    def __init__(self, path, add_undirected_edges, add_self_loops, mode, vel_history):
        super().__init__(path, add_undirected_edges, add_self_loops)
        self._mode = mode
        self._vel_history = vel_history
    
    def _load_data(self, path):
        """
            The resulting dataset has dimensions [num_trajectories, num_timesteps, (qs, dqs, ps, accs)]

        """
        data = np.load(path, allow_pickle=True)
        state = data['state_trajectories']
        config = data['config']
        self._dt = config['dt']
        # Masses
        self._m = jnp.array([config['m1'], config['m2']])
        # Absolute position
        self._qs = jnp.stack((state[:,:,0],  # q1
                        state[:,:,2]), # q2
                        axis=-1)
        # Relative positions
        self._dqs = jnp.expand_dims(self._qs[:,:,1] - self._qs[:,:,0], axis=-1)
        # Conjugate momenta
        self._ps = jnp.stack((state[:,:,1],  # p1
                        state[:,:,3]), # p2
                        axis=-1)
        # Velocities
        self._vs = self._ps / self._m
        # Accelerations
        self._accs = jnp.diff(self._vs, axis=1) / self._dt
        final_acc = jnp.expand_dims(self._accs[:,-1,:], axis=1) # duplicate final acceleration
        self._accs = jnp.concatenate((self._accs, final_acc), axis=1) # add copy of final acceleration to end of accs
        data = jnp.concatenate((self._qs, self._dqs,self._ps, self._accs), axis=-1)
        data = jax.lax.stop_gradient(data)

        self._data = data
    
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

        self._norm_stats = norm_stats
    
    def _setup_graph_params(self):
        self.n_node = jnp.array([jnp.shape(self._qs)[-1]])
        self.n_edge = jnp.array([jnp.shape(self._dqs)[-1]])
        self.senders = jnp.array([0])
        self.receivers = jnp.array([1])
    
    @jax.jit
    def get_graph(self, traj_idx, t) -> jraph.GraphsTuple:
        """ Need to make sure t > self._vel_history! """
        match self._mode:
            case 'acceleration':
                vs_history = []                
                [vs_history.append(self._vs[traj_idx, t-k]) for k in reversed(range(self._vel_history))]
                vs_history = jnp.asarray(vs_history).T
                # Node features are current position, velocity history, current velocity
                nodes = jnp.column_stack((self._qs[traj_idx, t], vs_history))
                # Edge features are relative positions
                edges = self._dqs[traj_idx, t].reshape((-1,1))
                # Global features are time, q0, v0
                global_context = jnp.concatenate((jnp.array([t]), self._qs[traj_idx, 0], self._vs[traj_idx, 0])).reshape(-1,1)
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
        aux_data = (self._path, self._add_undirected_edges, self._add_self_loops, self._mode, self._vel_history, self._data, self._norm_stats, self._qs, self._dqs, self._ps, self._vs, self._accs, self._m, self._dt) # static
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        obj = object.__new__(DMSDGraphBuilder)
        obj._path                   = aux_data[0]
        obj._add_undirected_edges   = aux_data[1]
        obj._add_self_loops         = aux_data[2]
        obj._mode                   = aux_data[3]
        obj._vel_history            = aux_data[4]
        obj._data                   = aux_data[5]
        obj._norm_stats             = aux_data[6]
        obj._qs                     = aux_data[7]
        obj._dqs                    = aux_data[8]
        obj._ps                     = aux_data[9]
        obj._vs                     = aux_data[10]
        obj._accs                   = aux_data[11]
        obj._m                      = aux_data[12]
        obj._dt                     = aux_data[13]
        obj._setup_graph_params()
        return obj