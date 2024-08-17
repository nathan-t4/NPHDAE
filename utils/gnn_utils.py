import jax
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from typing import Dict, Any
from functools import partial
       

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

def incidence_matrices_from_graph(graph, edge_types=None):
    ''' Return the incidence matrices of the given graph: (AC, AR, AL, AV, AI) '''
    AC = []
    AR = []
    AL = []
    AV = []
    AI = []
    A = [AC, AR, AL, AV, AI]

    for i in range(len(graph.edges)):
        label = edge_types[i]
        sender_idx = graph.senders[i]
        receiver_idx = graph.receivers[i]
        Ai = np.zeros((len(graph.nodes)))
        Ai[sender_idx] = -1
        Ai[receiver_idx] = 1
        A[label].append(Ai)

    A = [jnp.array(a).T if len(a) > 0 else None for a in A]
    return A

def graph_from_incidence_matrices(A):
    AC, AR, AL, AV, AI = A
    senders = []
    receivers = []
    for a in A:
        si = jnp.where(a == 1)[0]
        ri = jnp.where(a == -1)[0]
        # If a is all zeros then do not append
        [senders.append(s) for s in si]
        [receivers.append(r) for r in ri]

    senders = jnp.array(senders).squeeze()
    receivers = jnp.array(receivers).squeeze()

    return senders, receivers


def get_system_config(AC, AR, AL, AV, AI, Alambda=None):
    num_nodes = len(AC)
    num_capacitors = 0 if (AC == 0.0).all() else len(AC.T)
    num_inductors = 0 if (AL == 0.0).all() else len(AL.T)
    num_resistors = 0 if (AR == 0.0).all() else len(AR.T)
    num_volt_sources = 0 if (AV == 0.0).all() else len(AV.T)
    num_cur_sources = 0 if (AI == 0.0).all() else len(AI.T)
    state_dim = num_capacitors + num_inductors + num_nodes + num_volt_sources

    E = jax.scipy.linalg.block_diag(AC, jnp.eye(num_inductors), jnp.zeros((num_capacitors, num_nodes)), jnp.zeros((num_volt_sources, num_volt_sources)))

    J = jnp.zeros((state_dim, state_dim)) 

    # first row of equations of J 
    J = J.at[0:num_nodes, num_nodes:(num_nodes + num_inductors)].set(-AL) 
    J = J.at[
        0:(num_nodes),
        (num_nodes + num_inductors + num_capacitors)::
        ].set(-AV)

    # second row of equations of J
    J = J.at[num_nodes:(num_nodes + num_inductors),
            0:num_nodes].set(AL.transpose())
    
    # Final row of equations of J
    J = J.at[(num_nodes + num_inductors + num_capacitors)::,
            0:num_nodes].set(AV.transpose()) 
    
    g = lambda e : (AR.T @ e) / 1.0 

    def r(z):
        g = lambda e : (AR.T @ e) / 1.0 

        e = z[0:num_nodes]
        uc = z[
            num_nodes+num_inductors :
            num_nodes+num_inductors+num_capacitors
        ]

        curr_through_resistors = jnp.linalg.matmul(AR, g(e))
        charge_constraint = jnp.matmul(AC.T, e) - uc

        diss = jnp.zeros((state_dim,))
        diss = diss.at[0:num_nodes].set(curr_through_resistors)
        diss = diss.at[(num_nodes + num_inductors):(num_nodes + num_inductors + num_capacitors)].set(charge_constraint)

        return diss

    B = jnp.zeros((state_dim, num_cur_sources + num_volt_sources))
    B = B.at[0:num_nodes, 0:num_cur_sources].set(-AI)
    B = B.at[(num_nodes + num_inductors + num_capacitors):, num_cur_sources:].set(-jnp.eye(num_volt_sources))

    diff_indices = jnp.arange(num_capacitors+num_inductors)
    alg_indices = jnp.arange(
            num_capacitors+num_inductors, 
            num_capacitors+num_inductors+num_nodes+num_volt_sources
            )
    num_diff_vars = len(diff_indices)
    num_alg_vars = len(alg_indices)

    config = {
        'AC': AC,
        'AL': AL,
        'AR': AR,
        'AV': AV,
        'AI': AI,
        'num_nodes': num_nodes,
        'num_capacitors': num_capacitors,
        'num_inductors': num_inductors,
        'num_resistors': num_resistors,
        'num_volt_sources': num_volt_sources,
        'num_cur_sources': num_cur_sources,
        'state_dim': state_dim,
        'E': E,
        'J': J,
        'r': r,
        'B': B,
        'diff_indices': diff_indices,
        'alg_indices': alg_indices,
        'num_diff_vars': num_diff_vars,
        'num_alg_vars': num_alg_vars,
        'is_k': False,
    }

    return config

def get_system_k_config(AC, AR, AL, AV, AI, Alambda):
    num_nodes = len(AC)
    num_capacitors = 0 if (AC == 0.0).all() else len(AC.T)
    num_inductors = 0 if (AL == 0.0).all() else len(AL.T)
    num_resistors = 0 if (AR == 0.0).all() else len(AR.T)
    num_volt_sources = 0 if (AV == 0.0).all() else len(AV.T)
    num_cur_sources = 0 if (AI == 0.0).all() else len(AI.T)
    num_lamb = len(Alambda.T)
    state_dim = num_capacitors + num_inductors + num_nodes + num_volt_sources+num_lamb

    E = jax.scipy.linalg.block_diag(AC, jnp.eye(num_inductors), jnp.zeros((num_capacitors, num_nodes)), jnp.zeros((num_volt_sources, num_volt_sources)), jnp.zeros((num_lamb, num_lamb)))

    J = jnp.zeros((state_dim, state_dim)) 

    # first row of equations of J 
    J = J.at[0:num_nodes, num_nodes:(num_nodes + num_inductors)].set(-AL) 
    J = J.at[
        0:(num_nodes),
        (num_nodes + num_inductors + num_capacitors):(num_nodes + num_inductors + num_capacitors + num_volt_sources)
        ].set(-AV)
    J = J.at[
        0:num_nodes,
        (num_nodes + num_inductors + num_capacitors + num_volt_sources):,
        ].set(-Alambda)

    # second row of equations of J
    J = J.at[num_nodes:(num_nodes + num_inductors),
            0:num_nodes].set(AL.transpose())
    
    # Final row of equations of J
    J = J.at[(num_nodes + num_inductors + num_capacitors)::,
            0:num_nodes].set(AV.transpose()) 
    
    J = J.at[(num_nodes + num_inductors + num_capacitors + num_volt_sources):,
            0:num_nodes].set(-Alambda.transpose())
    
    g = lambda e : (AR.T @ e) / 1.0 

    def r(z):
        g = lambda e : (AR.T @ e) / 1.0 

        e = z[0:num_nodes]
        uc = z[
            num_nodes+num_inductors :
            num_nodes+num_inductors+num_capacitors
        ]

        curr_through_resistors = jnp.linalg.matmul(AR, g(e))
        charge_constraint = jnp.matmul(AC.T, e) - uc

        diss = jnp.zeros((state_dim,))
        diss = diss.at[0:num_nodes].set(curr_through_resistors)
        diss = diss.at[(num_nodes + num_inductors):(num_nodes + num_inductors + num_capacitors)].set(charge_constraint)

        return diss

    B = jnp.zeros((state_dim, num_cur_sources + num_volt_sources))
    B = B.at[0:num_nodes, 0:num_cur_sources].set(-AI)
    B = B.at[(num_nodes + num_inductors + num_capacitors):, num_cur_sources:].set(-jnp.eye(num_volt_sources))

    diff_indices = jnp.arange(num_capacitors+num_inductors)
    alg_indices = jnp.arange(
            num_capacitors+num_inductors, 
            num_capacitors+num_inductors+num_nodes+num_volt_sources+num_lamb
            )
    num_diff_vars = len(diff_indices)
    num_alg_vars = len(alg_indices)

    config = {
        'AC': AC,
        'AL': AL,
        'AR': AR,
        'AV': AV,
        'AI': AI,
        'num_nodes': num_nodes,
        'num_capacitors': num_capacitors,
        'num_inductors': num_inductors,
        'num_resistors': num_resistors,
        'num_volt_sources': num_volt_sources,
        'num_cur_sources': num_cur_sources,
        'state_dim': state_dim, 
        'E': E,
        'J': J,
        'r': r,
        'B': B,
        'diff_indices': diff_indices,
        'alg_indices': alg_indices,
        'num_diff_vars': num_diff_vars,
        'num_alg_vars': num_alg_vars,
        'is_k': True,
    }

    return config

def get_J_matrix(system_config):
    state_dim = system_config['state_dim']
    num_nodes = system_config['num_nodes']
    num_capacitors = system_config['num_capacitors']
    num_inductors = system_config['num_inductors']
    AL = system_config['AL']
    AV = system_config['AV']

    J = jnp.zeros((state_dim, state_dim)) 

    # first row of equations of J 
    J = J.at[0:num_nodes, num_nodes:(num_nodes + num_inductors)].set(-AL) 
    J = J.at[
        0:(num_nodes),
        (num_nodes + num_inductors + num_capacitors)::
        ].set(-AV)

    # second row of equations of J
    J = J.at[num_nodes:(num_nodes + num_inductors),
            0:num_nodes].set(AL.transpose())
    
    # Final row of equations of J
    J = J.at[(num_nodes + num_inductors + num_capacitors)::,
            0:num_nodes].set(AV.transpose()) 
    
    return J

def get_E_matrix(system_config):
    num_nodes = system_config['num_nodes']
    num_capacitors = system_config['num_capacitors']
    num_inductors = system_config['num_inductors']
    num_volt_sources = system_config['num_volt_sources']
    AC = system_config['AC']
    E = jax.scipy.linalg.block_diag(AC, jnp.eye(num_inductors), jnp.zeros((num_capacitors, num_nodes)), jnp.zeros((num_volt_sources, num_volt_sources)))
    return E

def get_B_bar_matrix(system_config):
    num_capacitors = system_config['num_capacitors']
    num_inductors = system_config['num_inductors']
    num_nodes = system_config['num_nodes']
    num_volt_sources = system_config['num_volt_sources']
    num_cur_sources = system_config['num_cur_sources']
    num_lamb = system_config['num_lamb']
    state_dim = num_capacitors+num_inductors+num_nodes+num_volt_sources+num_lamb
    AI = system_config['AI']

    B_bar = jnp.zeros((state_dim, num_cur_sources+num_volt_sources))

    B_bar.at[0:num_nodes, 0:num_cur_sources].set(-AI)

    B_bar.at[
        num_capacitors+num_inductors+num_nodes : num_capacitors+num_inductors+num_nodes+num_volt_sources,
        num_cur_sources : num_cur_sources+num_volt_sources
        ].set(-jnp.eye(num_volt_sources))
    
    return B_bar