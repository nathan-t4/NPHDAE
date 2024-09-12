import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from typing import Dict, Any
from functools import partial

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
        Ai = jnp.zeros((len(graph.nodes)))
        Ai = Ai.at[sender_idx].set(-1)
        Ai = Ai.at[receiver_idx].set(1)
        A[label].append(Ai)

    A = [jnp.array(a).T if len(a) > 0 else None for a in A]
    return A

def fill_in_incidence_matrix(A):
    # Fill in the missing ground node (first row) so that each column
    # of the incidence matrix has one (1) and one (-1) entry.
    gnd_row = []
    for col in A.T:
        if len(jnp.where(col == 1.0)[0]) == 0 and len(jnp.where(col == -1.0)[0]) == 1:
            gnd_row.append(1.0)
        elif len(jnp.where(col == -1.0)[0]) == 0 and len(jnp.where(col == 1.0)[0]) == 1:
            gnd_row.append(-1.0)
        else:
            gnd_row.append(0.0)
    gnd_row = jnp.array(gnd_row).reshape(1,-1)
    return jnp.concatenate((gnd_row, A))

def graph_from_incidence_matrices(A):
    AC, AR, AL, AV, AI = A
    new_A = [fill_in_incidence_matrix(a) for a in A]    
    senders = []
    receivers = []
    for a in new_A:
        si_row, si_col = jnp.where(a == 1)
        ri_row, ri_col = jnp.where(a == -1)
        # If a is all zeros then do not append            
        [senders.append(s) for s in si_row]
        [receivers.append(r) for r in ri_row]

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

    # diff_indices = jnp.arange(num_capacitors+num_inductors)
    # alg_indices = jnp.arange(
    #         num_capacitors+num_inductors, 
    #         num_capacitors+num_inductors+num_nodes+num_volt_sources
    #         )
    
    diff_indices, alg_indices = get_diff_and_alg_indices(E)
    alg_eq_indices = get_alg_eq_indices(E)
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
        'alg_eq_indices': alg_eq_indices,
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
    state_dim = num_capacitors + num_inductors + num_nodes + num_volt_sources + num_lamb
    
    E = jnp.zeros((state_dim, state_dim))
    E = E.at[0:num_nodes, 0:num_capacitors].set(AC)
    E = E.at[num_nodes:num_nodes+num_inductors, num_capacitors:num_capacitors+num_inductors].set(jnp.eye(num_inductors))

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
    J = J.at[(num_nodes + num_inductors + num_capacitors) :
            (num_nodes + num_inductors + num_capacitors + num_volt_sources),
            0:num_nodes].set(AV.transpose()) 
    
    J = J.at[(num_nodes + num_inductors + num_capacitors + num_volt_sources):,
            0:num_nodes].set(Alambda.transpose())
    

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
    B = B.at[(num_nodes + num_inductors + num_capacitors) :
             (num_nodes + num_inductors + num_capacitors + num_volt_sources), 
             num_cur_sources : (num_cur_sources + num_volt_sources)
             ].set(-jnp.eye(num_volt_sources))

    diff_indices, alg_indices = get_diff_and_alg_indices(E)
    alg_eq_indices = get_alg_eq_indices(E)
    num_diff_vars = len(diff_indices)
    num_alg_vars = len(alg_indices)

    config = {
        'AC': AC,
        'AR': AR,
        'AL': AL,
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
        'alg_eq_indices': alg_eq_indices,
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

    B_bar = B_bar.at[0:num_nodes, 0:num_cur_sources].set(-AI)

    B_bar = B_bar.at[
        num_capacitors+num_inductors+num_nodes : num_capacitors+num_inductors+num_nodes+num_volt_sources,
        num_cur_sources : num_cur_sources+num_volt_sources
        ].set(-jnp.eye(num_volt_sources))
    
    return B_bar

def get_B_hats(system_configs, Alambda):
    num_subsystems = len(system_configs)
    num_capacitors = [cfg['num_capacitors'] for cfg in system_configs]
    num_inductors = [cfg['num_inductors'] for cfg in system_configs]
    num_volt_sources = [cfg['num_volt_sources'] for cfg in system_configs]
    num_nodes = [cfg['num_nodes'] for cfg in system_configs]
    state_dims = [cfg['state_dim'] for cfg in system_configs]

    Alambdas = [
        Alambda[sum(num_nodes[:i]) : sum(num_nodes[:i+1])]
        for i in range(num_subsystems)
        ]
    
    num_lamb = len(Alambda.T)

    system_k_idx = -1

    # TODO: Define B_hat in get_system_config in gnn_utils
    B_hats = []
    for i, cfg in enumerate(system_configs):            
        # Find the index of the k-th system
        if cfg['is_k']:
            if system_k_idx > 0:
                raise ValueError(f"Last system has already been set to {system_k_idx}. Make sure there is only one subsystem with subsystem_config['last_system'] = True")
            else:
                system_k_idx = i
                B_hat_k = jnp.concatenate((
                    jnp.zeros((state_dims[i]-num_lamb, num_lamb)), jnp.eye(num_lamb)
                ))
                B_hats = [*B_hats, B_hat_k]
        else:
            B_hat_i = jnp.concatenate((
                Alambdas[i], 
                jnp.zeros((num_inductors[i]+num_capacitors[i]+num_volt_sources[i], num_lamb))
            ))
            B_hats = [*B_hats, B_hat_i]

    return B_hats

def get_system_k_idx(system_configs):
    system_k_idx = -1
    for i, cfg in enumerate(system_configs):            
        # Find the index of the k-th system
        if cfg['is_k']:
            if system_k_idx > 0:
                raise ValueError(f"Last system has already been set to {system_k_idx}. Make sure there is only one subsystem with subsystem_config['last_system'] = True")
            else:
                system_k_idx = i
    
    assert(system_k_idx != -1), \
            "k-th system index has not been set. \
            Make sure that one system config in self.system_configs has cfg['is_k'] = True!"
    
    return system_k_idx

def get_diff_and_alg_indices(E):
    diff_indices = jnp.where(jnp.array([(E[:, col] != 0.0).any() for col in range(E.shape[1])]))[0]
    alg_indices = jnp.where(jnp.array([(E[:, col] == 0.0).all() for col in range(E.shape[1])]))[0]
    return diff_indices, alg_indices

def get_alg_eq_indices(E):
    alg_eq_indices = jnp.where(jnp.array([(E[row, :] == 0.0).all() for row in range(E.shape[0])]))[0]
    return alg_eq_indices

def nonzero_columns(matrix):
  """Returns the nonzero columns of a matrix.

  Args:
    matrix: A JAX array representing the input matrix.

  Returns:
    A JAX array containing the nonzero columns of the input matrix.
  """

  # Get the indices of nonzero elements
  nonzero_indices = jnp.nonzero(matrix)

  # Extract the column indices
  column_indices = jnp.unique(nonzero_indices[1])

  # Extract nonzero columns using the mask
  nonzero_columns = matrix[:, column_indices]

  return nonzero_columns