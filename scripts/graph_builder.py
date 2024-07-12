import jraph
import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
from jax.tree_util import register_pytree_node_class
from copy import deepcopy
from functools import partial
from typing import Sequence, Tuple
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
    
    def get_control(self, trajs, ts):
        raise NotImplementedError
    
    def get_pred_data(self, graph):
        raise NotImplementedError
    
    def get_exp_data(self, trajs, ts):
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
        self._V2 = self._Q1 / self.C
        self._V3 = self._Q3 / self.C_prime
        self._H = 0.5 * (self._Q1**2 / self.C + self._Q3**2 / self.C_prime + self._Phi1**2 / self.L)
        self._control = jnp.array(u).squeeze()

        self.edge_idxs = np.array([[0,2]])
        self.node_idxs = None
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
        pred_H = (graph.globals).squeeze()
        return (pred_Q1, pred_Phi1, pred_Q3, pred_H) 

    def get_exp_data(self, trajs, ts) -> Tuple:
        return (self._Q1[trajs, ts], self._Phi1[trajs, ts], self._Q3[trajs, ts], self._H[trajs, ts])
    
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

        norm_stats.Q3 = ml_collections.ConfigDict({
            'mean': jnp.mean(self._Q3),
            'std': jnp.std(self._Q3),
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
        self.senders = jnp.array([0, 1, 0])
        self.receivers = jnp.array([1, 2, 2])

    def get_graph_from_state(self, state, control=None, system_params=None, set_nodes=False, set_ground_and_control=False, nodes=None, globals=None) -> jraph.GraphsTuple:
        # Q1 = state[0]
        # Phi1 = state[1]
        # Q3 = state[2]
        Q1 = state[0]
        Q3 = state[1]
        Phi1 = state[2]
        nodes = nodes
        if set_nodes:
            V2 = Q1 / system_params['C']
            V3 = Q3 / system_params['C_prime']
            nodes = jnp.array([[0], [V2], [V3]])
        if set_ground_and_control:
            nodes = jnp.concatenate((jnp.array([[0]]), nodes[1:]), axis=0)

        edges = jnp.array([[Q1, 0], [Phi1, 1], [Q3, 0]])
        n_node = jnp.array([3])
        n_edge = jnp.array([3])
        senders = jnp.array([0, 1, 0])
        receivers = jnp.array([1, 2, 2])
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
    
    def get_graph(self, traj_idx, t) -> jraph.GraphsTuple:
        nodes = jnp.array([[0], [self._V2[traj_idx, t]], [self._V3[traj_idx, t]]])
        edges = jnp.array([[self._Q1[traj_idx, t], 0], 
                           [self._Phi1[traj_idx, t], 1], 
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
                          [0],
                          self._V2[traj_idx, t],
                          self._V3[traj_idx, t]])
    
    def get_state_batch(self, traj_idxs, t0s) -> jnp.ndarray:
        def f(carry, idxs):
            return carry, self.get_state(*idxs)
        
        _, states = jax.lax.scan(f, None, (traj_idxs, t0s))
    
    def tree_flatten(self):
        children = ()
        aux_data = (self._dt, self.C, self.C_prime, self.L, self.system_params, self._Q1, self._Phi1, self._Q3, self._V2, self._V3, self._H, self._control, self._num_trajectories, self._num_timesteps, self._num_states)
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
        obj._V2                   = aux_data[8]
        obj._V3                   = aux_data[9]
        obj._H                    = aux_data[10]
        obj._control              = aux_data[11]
        obj._num_trajectories     = aux_data[12]
        obj._num_timesteps        = aux_data[13]
        obj._num_states           = aux_data[14]

        obj._setup_graph_params()
        return obj
    
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
    

# @register_pytree_node_class # TODO
class PowerGridGraphBuilder(GraphBuilder):
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