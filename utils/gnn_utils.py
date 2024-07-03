import jax
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from typing import Dict, Any

def get_edge_idxs(name):
    match name:
        case 'LC1':
            return np.array([[0,2]])
        case 'LC2':
            return np.array([[0]])
        case 'CoupledLC':
            return np.array([[0,2,3]])
        case 'Alternator': 
            return np.array([[0,1,2,3,4,5], [6,None,None,None,None,None]])
        case _:
            raise NotImplementedError(f"Edge indices not set for system {name}")
        
def get_pH_matrices(name):
    return (get_J(name), get_R(name), get_g(name))

def get_J(name):
    match name:
        case 'LC1':
            return jnp.array([[0, 1, 0],
                              [-1, 0, 1],
                              [0, -1, 0]])
        case 'LC2':
            return jnp.array([[0, 1],
                              [-1, 0]])
        case 'CoupledLC':
            return jnp.array([[0, 1, 0, 0, 0],
                              [-1, 0, 1, 0, 0],
                              [0, -1, 0, 0, -1],
                              [0, 0, 0, 0, 1],
                              [0, 0, 1, -1, 0]])
        case 'Alternator':
            return jnp.array([[0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, -1],
                              [0, 0, 0, 0, 0, 0, 1, 0]])
        case _:
            raise NotImplementedError(f"J matrix not set for system {name}")
        
def get_R(name):
    match name:
        case 'LC1':
            return jnp.zeros((3,3))
        case 'LC2':
            return jnp.zeros((2,2))
        case 'CoupledLC':
            return jnp.zeros((5,5))
        case 'Alternator':
            return jax.scipy.linalg.block_diag(jnp.array([0])) # TODO: depends on traj_idx (rm and rr)
        case _:
            raise NotImplementedError(f"R matrix not set for system {name}")
        
def get_g(name):
    match name:
        case 'LC1':
            return jnp.array([[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, -1]])
        case 'LC2':
            # return jnp.array([[0, 0],
            #                   [0,-1]]) 
            return jnp.zeros((2,2))
        case 'CoupledLC':
            return jnp.zeros((5,5))
        case 'Alternator': 
            return jnp.zeros((8,2))
        case _:
            raise NotImplementedError(f"g matrix not set for system {name}")