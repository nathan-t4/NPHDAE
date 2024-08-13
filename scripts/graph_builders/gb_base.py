import jraph
import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
from functools import partial
from typing import Sequence, Tuple
from jax.tree_util import register_pytree_node_class


class GraphBuilder():
    def __init__(self, path, AC, AR, AL, AV, AI, add_undirected_edges, add_self_loops):
        self._path = path
        assert (len(AC) == len(AR) == len(AL) == len(AV) == len(AI))
        self.AC = AC
        self.AR = AR
        self.AL = AL
        self.AV = AV
        self.AI = AI

        self.num_nodes = len(AC)
        self.num_capacitors = 0 if (AC == 0.0).all() else len(AC.T)
        self.num_resistors =  0 if (AR == 0.0).all() else len(AR.T)
        self.num_inductors =  0 if (AL == 0.0).all() else len(AL.T)
        self.num_volt_sources = 0 if (AV == 0.0).all() else len(AV.T)
        self.num_cur_sources = 0 if (AI == 0.0).all() else len(AI.T)
        self._num_states = self.num_capacitors + self.num_inductors + self.num_nodes + self.num_volt_sources
        
        self.n_node = jnp.array([self.num_nodes]) # + 1 for ground node
        self.n_edge = jnp.array([self.num_capacitors + self.num_inductors + self.num_resistors + self.num_volt_sources + self.num_cur_sources])
        
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