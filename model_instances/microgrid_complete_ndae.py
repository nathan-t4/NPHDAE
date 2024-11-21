import sys, os
# sys.path.append('../')
from models.ph_dae import PHDAE
from models.composite_ph_dae import CompositePHDAE
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import jax
from helpers.model_factory import get_model_factory
import json

from plotting.common import load_config_file, load_dataset, load_model, load_metrics, predict_trajectory, compute_traj_err, compute_g_vals_along_traj
import argparse
from models.ph_dae import PHDAE
import pickle
from tqdm import tqdm
from model_instances.helper import *

def get_system_params(num_dgu):
    num_tl = int(0.5 * (num_dgu * (num_dgu - 1)))
    num_capacitors = num_dgu
    num_inductors = num_dgu + num_tl
    num_nodes = 3 * num_dgu + 3 * num_tl
    num_volt_sources = num_dgu
    num_couplings = 2 * num_tl

    v = [1.00] * num_dgu
    R_tl = [0.05] * num_tl
    L_tl = [1.8e-3] * num_tl
    i_load = [0.9, 0.8, 1.1, 1.3, 0.5, 0.2, 0.9, 0.7, 1.0, 0.3, 0.2, 0.6, 0.5, 0.3, 0.8, 1.2]
    i_load = i_load[:num_dgu]

    system_params = {
        'R': [0.2] * num_dgu + R_tl,
        'L': [1.8e-3] * num_dgu + L_tl,
        'C': [2.2e-3] * num_dgu,
        'i_load': i_load,
        'v': v,
    }

    return system_params

def dc_microgrid_ndae(exp_file_name, num_dgu=3, z0=None, T=None, control=None, plot=True, seed=42):
    ph_dae_list = []
    params_list = []
    exp_file_name = [exp_file_name] * num_dgu

    num_tl = int(0.5 * (num_dgu * (num_dgu - 1)))
    num_capacitors = num_dgu
    num_inductors = num_dgu + num_tl
    num_nodes = 3 * num_dgu + 3 * num_tl
    num_volt_sources = num_dgu
    num_couplings = 2 * num_tl

    # R_tl = 0.05
    # L_tl = 1.8e-3
    # R_tl = 2
    # L_tl = 1.8

    # R_tl = 1.0
    # L_tl = 1.0

    v = [1.00] * num_dgu
    # R_tl = [R_tl] * num_tl
    # L_tl = [L_tl] * num_tl

    key = jax.random.key(seed)
    key, Rkey, Lkey, ikey = jax.random.split(key, 4)
    R_tl = jax.random.uniform(Rkey, shape=(num_tl,), minval=0.1, maxval=2.0)
    L_tl = jax.random.uniform(Lkey, shape=(num_tl,), minval=0.1, maxval=2.0)
    i_load = jax.random.uniform(ikey, shape=(num_dgu,), minval=0.1, maxval=1.0)

    # R_dgu = 0.2
    # L_dgu = 1.8e-3
    # C_dgu = 2.2e-3

    # R_dgu = 1.0
    # L_dgu = 1.0
    # C_dgu = 1.0

    Rs = []
    Ls = []
    Cs = []    

    assert len(exp_file_name) == num_dgu

    for i in range(num_dgu):
        ndgu_model, ndgu_params, ndgu_model_setup = add_ndgu(exp_file_name[i], i_load[i], v[i])
        ph_dae_list.append(ndgu_model)
        params_list.append(ndgu_params)
        R, L, C = ndgu_model_setup['R'], ndgu_model_setup['L'], ndgu_model_setup['C']
        Rs.append(R)
        Ls.append(L)
        Cs.append(C)

    for i in range(num_tl):
        tl_model, tl_params = add_transmission_line(R_tl[i], L_tl[i])
        ph_dae_list.append(tl_model)
        params_list.append(tl_params)
        Rs.append(R_tl[i])
        Ls.append(L_tl[i])
    
    system_params = {
        'R': Rs, # [R_dgu] * num_dgu + R_tl,
        'L': Ls, # [L_dgu] * num_dgu + L_tl,
        'C': Cs, # [C_dgu] * num_dgu,
        'i_load': i_load,
        'v': v,
    }

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

    composite_ndae = CompositePHDAE(ph_dae_list, A_lambda)

    if z0 is None:
        key, subkey = jax.random.split(key)
        x0 = jnp.zeros(num_capacitors+num_inductors)
        x0 = 0.5 * (jax.random.uniform(subkey, num_capacitors+num_inductors) * 2 - 1)
        y0 = jnp.zeros(num_nodes+num_volt_sources+num_couplings)
        z0 = jnp.concatenate((x0, y0))

    if T is None:
        dt = 1e-3
        sim_time = 2
        Tf = int(sim_time/dt)
        T = jnp.linspace(0, sim_time, Tf+1)[:Tf]

    if control is None:
        pass # TODO

    # sol = composite_ndae.solve(z0, T, params_list=params_list)
    sol = composite_ndae.solve_one_timestep(z0, T, params_list=params_list, control=control)

    if plot:
        from plotting.common import compute_g_vals_along_traj
        gnorm, gval = compute_g_vals_along_traj(composite_ndae.solver.g, params_list, sol, T, num_diff_vars=num_capacitors+num_inductors)

        phndae_color = (12/255, 212/255, 82/255)
        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(111)
        ax.plot(T, gnorm, color=phndae_color, linewidth=5)

        plt.savefig(f'microgrid_{num_dgu}_ndae_gnorm.png')

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

        # exp_traj = generate_dataset(num_dgu=num_dgu, ntimesteps=Tf, dt=dt, z0=z0_adjusted, plot=False, system_params=system_params)['state_trajectories'][0]
        exp_traj = jnp.zeros_like(sol)

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
        ax[0,0].plot(T, 
                    exp_traj[:,jnp.array(exp_diff_states_idx[:num_capacitors])], 
                    label=exp_diff_labels[:num_capacitors],
                    color='black')
        ax[0,0].plot(T, 
                    sol[:,jnp.array(diff_states_idx[:num_capacitors])], 
                    label=diff_labels[:num_capacitors],
                    color=phndae_color,
                    linestyle='--')

        ax[0,1].plot(T, 
                    exp_traj[:,jnp.array(exp_diff_states_idx[num_capacitors:])], 
                    label=exp_diff_labels[num_capacitors:],
                    color='black')
        ax[0,1].plot(T,
                    sol[:,jnp.array(diff_states_idx[num_capacitors:])],
                    label=diff_labels[num_capacitors:],
                    color=phndae_color,
                    linestyle='--')

        ax[1,0].plot(T, 
                    exp_traj[:,jnp.array(exp_alg_states_idx[:exp_num_nodes])], 
                    label=exp_alg_labels[:exp_num_nodes],
                    color='black')
        ax[1,0].plot(T,
                    sol[:,jnp.array(alg_states_idx[:num_nodes])],
                    label=alg_labels[:num_nodes],
                    color=phndae_color,
                    linestyle='--')

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
        plt.savefig(f'microgrid_{num_dgu}_ndae_traj.png')
        plt.show()
        plt.close()

        # fig, ax = plt.subplots(2, 2)
        # ax[0,0].plot(T, exp_traj[:,jnp.array(exp_diff_states_idx[:num_capacitors])] - sol[:,jnp.array(diff_states_idx[:num_capacitors])])
        # ax[0,1].plot(T, exp_traj[:,jnp.array(exp_diff_states_idx[num_capacitors:])] - sol[:,jnp.array(diff_states_idx[num_capacitors:])])
        # plt.tight_layout()
        # plt.show()
        # plt.close()

        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(111)
        ax.plot(T, exp_traj[:,num_capacitors+num_inductors+1], color='black', linewidth=5)
        ax.plot(T, sol[:,num_capacitors+num_inductors+1], '--', color=phndae_color, linewidth=5)
        plt.savefig(f'microgrid_{num_dgu}_ndae_one_state.png')
        plt.close()
    
    return sol, composite_ndae, system_params

if __name__ == '__main__':
    num_dgu = 2
    # exp_file_name = '2024-10-22_12-25-00_phdae_dgu_user51'
    # exp_file_name = '2024-11-03_13-42-46_phdae_dgu_user_1'
    exp_file_name = 'dgu/2024-11-14_20-21-20_phdae_dgu'
    dc_microgrid_ndae(exp_file_name, num_dgu)
