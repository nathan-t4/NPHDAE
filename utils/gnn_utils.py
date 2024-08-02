import jax
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from typing import Dict, Any
from functools import partial

"""
    TODO: remove
"""

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
        

def fwd_solver(f, z_init):
    def cond_fun(carry):
        z_prev, z = carry
        return jnp.linalg.norm(z_prev - z) > 1e-5

    def body_fun(carry):
        _, z = carry
        return z, f(z)

    init_carry = (z_init, f(z_init))
    _, z_star = jax.lax.while_loop(cond_fun, body_fun, init_carry)
    return z_star


@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def fixed_point_layer(solver, f, params, x):
    z_star = solver(lambda z: f(params, x, z), z_init=jnp.zeros_like(x))
    return z_star

def fixed_point_layer_fwd(solver, f, params, x):
    z_star = fixed_point_layer(solver, f, params, x)
    return z_star, (params, x, z_star)

def fixed_point_layer_bwd(solver, f, res, z_star_bar):
    params, x, z_star = res
    _, vjp_a = jax.vjp(lambda params, x: f(params, x, z_star), params, x)
    _, vjp_z = jax.vjp(lambda z: f(params, x, z), z_star)
    return vjp_a(solver(lambda u: vjp_z(u)[0] + z_star_bar,
                      z_init=jnp.zeros_like(z_star)))

def get_nonzero_row_indices(array):
    def f(carry, row):
        nonzero_row = sum(jnp.abs(row))
        return carry, nonzero_row
    _, row_abs_sums = jax.lax.scan(f, None, array[:])
    nonzero_rows_mask = jnp.nonzero(row_abs_sums)
    return jnp.arange(len(array))[nonzero_rows_mask]

def get_zero_row_indices(array):
    def f(carry, row):
        nonzero_row = sum(jnp.abs(row))
        return carry, nonzero_row
    _, row_abs_sums = jax.lax.scan(f, None, array[:])
    zero_rows_mask = jnp.where(row_abs_sums == 0)
    return jnp.arange(len(array))[zero_rows_mask]
