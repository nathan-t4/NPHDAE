import jraph
import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
from functools import partial
from typing import Sequence, Tuple
from jax.tree_util import register_pytree_node_class


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