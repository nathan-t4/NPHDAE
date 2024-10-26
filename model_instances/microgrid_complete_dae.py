import sys
sys.path.append('../')
from models.ph_dae import PHDAE
from models.composite_ph_dae import CompositePHDAE
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from model_instances.helper import *


def dc_microgrid_dae(num_dgu, z0=None, T=None, system_params=None, plot=True):
    ph_dae_list = []
    params_list = []

    num_tl = int(0.5 * (num_dgu * (num_dgu - 1)))
    num_capacitors = num_dgu
    num_inductors = num_dgu + num_tl
    num_resistors = num_dgu + num_tl
    num_nodes = 3 * num_dgu + 3 * num_tl
    num_volt_sources = num_dgu
    num_cur_sources = num_dgu
    num_couplings = 2 * num_tl

    if system_params is None:
        R = [1.0] * num_resistors
        L = [1.0] * num_inductors
        C = [1.0] * num_capacitors
        i_load = [0.1] * num_cur_sources
        v_load = [1.0] * num_volt_sources
    else:
        R = system_params['R']
        L = system_params['L']
        C = system_params['C']    
        i_load = system_params['i_load']
        v = system_params['v']

    assert(len(R) == num_resistors)
    assert(len(L) == num_inductors)
    assert(len(C) == num_capacitors)
    assert(len(i_load) == num_cur_sources)
    assert(len(v) == num_volt_sources)

    # Distributed generation units
    for i in range(num_dgu):
        dgu, dgu_params = add_dgu(R[i], L[i], C[i], i_load[i], v[i])
        ph_dae_list.append(dgu)
        params_list.append(dgu_params)

    # Transmission lines
    for i in range(num_tl):
        tl, tl_params = add_transmission_line(R[num_dgu+i], L[num_dgu+i])
        ph_dae_list.append(tl)
        params_list.append(tl_params)


    # A_lambda_t = np.zeros((num_nodes,num_couplings))
    # A_lambda_t[2,0] = 1; A_lambda_t[9,0] = -1
    # A_lambda_t[5,1] = 1; A_lambda_t[11,1] = -1
    # A_lambda_t[2,2] = 1; A_lambda_t[12,2] = -1
    # A_lambda_t[8,3] = 1; A_lambda_t[14,3] = -1
    # A_lambda_t[5,4] = 1; A_lambda_t[15,4] = -1
    # A_lambda_t[8,5] = 1; A_lambda_t[17,5] = -1
    # A_lambda_t = jnp.array(A_lambda_t)

    # Coupling matrix
    A_lambda = np.zeros((num_nodes,num_couplings))
    k = 0
    for i in range(num_dgu - 1):
        base_dgu_idx = 3 * i + 2
        for j in range(num_dgu - i - 1):
            # base_dgu <-> base_tl
            tl_node_idx = 3 * num_dgu + 3 * (k//2)
            A_lambda[base_dgu_idx,k] = 1
            A_lambda[tl_node_idx,k] = -1
            k = k + 1
            # end_dgu <-> end_tl
            end_dgu_idx = 3 * (i + j + 1) + 2
            A_lambda[end_dgu_idx,k] = -1
            A_lambda[tl_node_idx+2,k] = 1
            k = k + 1

    A_lambda = jnp.array(A_lambda)

    # print("####################################")
    # print(A_lambda)
    # print("####################################")

    composite_dae = CompositePHDAE(ph_dae_list, A_lambda)

    if z0 is None:
        seed = 42
        key = jax.random.key(seed)
        x0 = jnp.zeros(num_capacitors+num_inductors)
        x0 = 0.5 * (jax.random.uniform(key, num_capacitors+num_inductors) * 2 - 1)
        y0 = jnp.zeros(num_nodes+num_volt_sources+num_couplings)
        z0 = jnp.concatenate((x0, y0))

    if T is None:
        dt = 1e-3
        sim_time = 2
        Tf = int(sim_time/dt)
        T = jnp.linspace(0, sim_time, Tf+1)[:Tf]

    assert(len(z0) == num_capacitors+num_inductors+num_nodes+num_volt_sources+num_couplings)

    sol = composite_dae.solve(z0, T, params_list=params_list)

    if plot:
        from plotting.common import compute_g_vals_along_traj
        gnorm, gval = compute_g_vals_along_traj(composite_dae.solver.g, params_list, sol, T, num_diff_vars=num_capacitors+num_inductors)

        phndae_color = (12/255, 212/255, 82/255)
        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(111)
        ax.plot(T, gnorm, color=phndae_color, linewidth=5)

        plt.savefig('dgu_pentagon_gnorm_new.png')

        from environments.microgrid_complete import generate_dataset
        x0 = z0[:num_capacitors+num_inductors+3*num_dgu]
        node_indices = jnp.arange(num_capacitors+num_inductors+3*num_dgu+1, num_capacitors+num_inductors+num_nodes, 3)
        # y0_adjusted = z0[num_capacitors+num_inductors+3*num_dgu:-num_couplings::3]
        jv_indices = jnp.arange(num_capacitors+num_inductors+num_nodes, num_capacitors+num_inductors+num_nodes+num_volt_sources)
        y0_adjusted = jnp.concatenate((z0[node_indices], z0[jv_indices]))
        z0_adjusted = jnp.concatenate((x0, y0_adjusted))

        # print("###############################################")
        # print(z0_adjusted.shape)
        # print(z0.shape)
        # print("###############################################")

        exp_traj = generate_dataset(num_dgu=num_dgu, ntimesteps=Tf, dt=dt, z0=z0_adjusted, plot=False)['state_trajectories'][0]

        def get_labels(num_nodes):
            diff_states_idx = np.arange(0,num_capacitors+num_inductors)
            alg_states_idx = np.arange(num_capacitors+num_inductors, num_capacitors+num_inductors+num_nodes+num_volt_sources)
            q_labels = [f'q{i}' for i in range(num_capacitors)]
            phi_labels = [f'phi{i}' for i in range(num_inductors)]
            e_labels = [f'e{i}' for i in range(num_nodes)]
            jv_labels = [f'jv{i}' for i in range(num_volt_sources)]
            diff_labels = np.concatenate((q_labels, phi_labels))
            alg_labels = np.concatenate((e_labels, jv_labels))
            return diff_states_idx, alg_states_idx, diff_labels, alg_labels

        exp_num_nodes = 3*num_dgu + num_tl
        exp_diff_states_idx, exp_alg_states_idx, exp_diff_labels, exp_alg_labels = get_labels(num_nodes=exp_num_nodes)
        diff_states_idx, alg_states_idx, diff_labels, alg_labels = get_labels(num_nodes=num_nodes)
        fig, ax = plt.subplots(2, 2)

        ax[0,0].set_title(r'$q$')
        ax[0,0].plot(T, 
                    exp_traj[:,jnp.array(exp_diff_states_idx[:num_capacitors])], 
                    label=exp_diff_labels[:num_capacitors],
                    color='black')
        ax[0,0].plot(T, 
                    sol[:,jnp.array(diff_states_idx[:num_capacitors])], 
                    label=diff_labels[:num_capacitors],
                    color=phndae_color,
                    linestyle='--')
        
        ax[0,1].set_title(r'$\phi$')
        ax[0,1].plot(T, 
                    exp_traj[:,jnp.array(exp_diff_states_idx[num_capacitors:])], 
                    label=exp_diff_labels[num_capacitors:],
                    color='black')
        ax[0,1].plot(T,
                    sol[:,jnp.array(diff_states_idx[num_capacitors:])],
                    label=diff_labels[num_capacitors:],
                    color=phndae_color,
                    linestyle='--')
        
        ax[1,0].set_title(r'$V$')
        ax[1,0].plot(T, 
                    exp_traj[:,jnp.array(exp_alg_states_idx[:exp_num_nodes])], 
                    label=exp_alg_labels[:exp_num_nodes],
                    color='black')
        ax[1,0].plot(T,
                    sol[:,jnp.array(alg_states_idx[:num_nodes])],
                    label=alg_labels[:num_nodes],
                    color=phndae_color,
                    linestyle='--')

        ax[1,1].set_title(r'$j_V$')
        ax[1,1].plot(T, 
                    exp_traj[:,jnp.array(exp_alg_states_idx[exp_num_nodes:])], 
                    label=exp_alg_labels[exp_num_nodes:],
                    color='black')
        ax[1,1].plot(T,
                    sol[:,jnp.array(alg_states_idx[num_nodes:])],
                    label=alg_labels[num_nodes:],
                    color=phndae_color,
                    linestyle='--')

        # ax[0,0].legend(loc='upper right')
        # ax[0,1].legend(loc='upper right')
        # ax[1,0].legend(loc='upper right')
        # ax[1,1].legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('dgu_pentagon_trajectory_new.png')
        plt.show()
        plt.close()

        fig, ax = plt.subplots(2, 2)
        ax[0,0].plot(T, exp_traj[:,jnp.array(exp_diff_states_idx[:num_capacitors])] - sol[:,jnp.array(diff_states_idx[:num_capacitors])])
        ax[0,1].plot(T, exp_traj[:,jnp.array(exp_diff_states_idx[num_capacitors:])] - sol[:,jnp.array(diff_states_idx[num_capacitors:])])
        plt.tight_layout()
        plt.show()
        plt.close()


        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(111)
        ax.plot(T, exp_traj[:,num_capacitors+num_inductors+1], color='black', linewidth=5)
        ax.plot(T, sol[:,num_capacitors+num_inductors+1], '--', color=phndae_color, linewidth=5)
        plt.savefig('dgu_pentagon_rollout_new.png')
        plt.close()

    return sol, composite_dae