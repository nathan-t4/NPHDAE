import sys, os
sys.path.append('../')
from models.ph_dae import PHDAE
from models.composite_ph_dae import CompositePHDAE
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from helpers.model_factory import get_model_factory
import json

from plotting.common import load_config_file, load_dataset, load_model, load_metrics, predict_trajectory, compute_traj_err, compute_g_vals_along_traj
import argparse
from models.ph_dae import PHDAE
import pickle
from tqdm import tqdm

def dc_microgrid_ndae(z0, T, exp_file_name, regularization_method='none', reg_param=0.0):
    ph_dae_list = []
    params_list = []

    # Load the DGU1 model
    # exp_file_name = '2024-08-06_18-40-56_train_phdae_dgu'
    # exp_file_name = '2024-09-20_12-24-41_train_phdae_dgu' # 1e-4
    # exp_file_name = '2024-09-23_19-45-01_train_phdae_dgu' # 1e-5
    # exp_file_name = '2024-10-03_12-51-32_train_phdae_dgu_1e-5_lamb_1e-3'
    # exp_file_name = '2024-09-30_19-51-40_train_phdae_dgu_1e-8' # 1e-8
    # exp_file_name = '2024-10-09_13-26-54_phdae_dgu_1e-5_scaled_g'
    # exp_file_name = '2024-10-09_18-53-46_phdae_dgu_1e-5_1e-1_lr1e-5' # with weight decay
    # exp_file_name = '2024-10-13_20-18-14_phdae_dgu_clipping_adamw' # train w/o q_func
    sacred_save_path = os.path.abspath(os.path.join('../cyrus_experiments/runs/', exp_file_name, '1'))

    config = load_config_file(sacred_save_path)
    model_setup = config['model_setup']
    model_setup['u_func_freq'] = 0.0
    model_setup['u_func_current_source_magnitude'] = 0.8 # 0.8
    model_setup['u_func_voltage_source_magnitude'] = 100.0 # 100.0
    model1 = get_model_factory(model_setup).create_model(jax.random.PRNGKey(0))

    # Load the "Run" json file to get the artifacts path
    run_file_str = os.path.abspath(os.path.join(sacred_save_path, 'run.json'))
    with open(run_file_str, 'r') as f:
        run = json.load(f)

    # Load the params for model 1
    artifacts_path = os.path.abspath(os.path.join(sacred_save_path, 'model_params.pkl'))
    with open(artifacts_path, 'rb') as f:
        params1 = pickle.load(f)

    ph_dae_list.append(model1.dae)
    params_list.append(params1)

    # Transmission line
    AC = jnp.array([[0.0], [0.0], [0.0]])
    AR = jnp.array([[-1.0], [1.0], [0.0]])
    AL = jnp.array([[0.0], [-1.0], [1.0]])
    AV = jnp.array([[0.0], [0.0], [0.0]])
    AI = jnp.array([[0.0], [0.0], [0.0]])

    R = 0.05
    L = 1.8e-3

    # R = 1.0
    # L = 1.0

    def r_func(delta_V, params=None):
        return delta_V / R

    def q_func(delta_V, params=None):
        return None

    def grad_H_func(phi, params=None):
        return phi / L

    def u_func(t, params):
        return jnp.array([])

    transmission_line = PHDAE(AC, AR, AL, AV, AI, grad_H_func, q_func, r_func, u_func)
    ph_dae_list.append(transmission_line)
    params_list.append(None)

    # Load the DGU2 model
    # exp_file_name = '2024-08-06_18-40-56_train_phdae_dgu'
    # exp_file_name = '2024-09-20_12-24-41_train_phdae_dgu' # 1e-4
    # exp_file_name = '2024-09-23_19-45-01_train_phdae_dgu' # 1e-5
    # exp_file_name = '2024-10-03_12-51-32_train_phdae_dgu_1e-5_lamb_1e-3'
    # exp_file_name = '2024-09-30_19-51-40_train_phdae_dgu_1e-8' # 1e-8
    # exp_file_name = '2024-10-09_13-26-54_phdae_dgu_1e-5_scaled_g'
    # exp_file_name = '2024-10-09_18-53-46_phdae_dgu_1e-5_1e-1_lr1e-5' # with weight decay
    # exp_file_name = '2024-10-13_20-18-14_phdae_dgu_clipping_adamw' # train w/o q_func
    sacred_save_path = os.path.abspath(os.path.join('../cyrus_experiments/runs/', exp_file_name, '1'))

    config = load_config_file(sacred_save_path)
    model_setup = config['model_setup']
    model_setup['u_func_freq'] = 0.0
    model_setup['u_func_current_source_magnitude'] = 1.1 # 1.1
    model_setup['u_func_voltage_source_magnitude'] = 100.0 # 100.0
    model2 = get_model_factory(model_setup).create_model(jax.random.PRNGKey(0))

    # Load the "Run" json file to get the artifacts path
    run_file_str = os.path.abspath(os.path.join(sacred_save_path, 'run.json'))
    with open(run_file_str, 'r') as f:
        run = json.load(f)

    # Load the params for model 1
    artifacts_path = os.path.abspath(os.path.join(sacred_save_path, 'model_params.pkl'))
    with open(artifacts_path, 'rb') as f:
        params2 = pickle.load(f)

    ph_dae_list.append(model2.dae)
    params_list.append(params2)

    # Build the composite microgrid model
    A_lambda = jnp.array([
        [0.0, 0.0], 
        [0.0, 0.0],
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 0.0],
        [0.0, -1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
    ])
    composite_ndae = CompositePHDAE(ph_dae_list, A_lambda, regularization_method, reg_param)

    # sol = composite_ndae.solve(z0, T, params_list=params_list)
    sol = composite_ndae.solve_one_timestep(z0, T, params_list=params_list)

    fig = plt.figure(figsize=(10, 20))

    ax1 = fig.add_subplot(631)
    ax1.plot(T, sol[:,0])
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel(r'$q_{1}$')

    ax1 = fig.add_subplot(632)
    ax1.plot(T, sol[:,1])
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel(r'$q_{2}$')

    ax1 = fig.add_subplot(633)
    ax1.plot(T, sol[:,2])
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel(r'$\phi_{1}$')

    ax1 = fig.add_subplot(634)
    ax1.plot(T, sol[:,3])
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel(r'$\phi_{2}$')

    ax1 = fig.add_subplot(635)
    ax1.plot(T, sol[:,4])
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel(r'$\phi_{3}$')

    ax1 = fig.add_subplot(636)
    ax1.plot(T, sol[:,5])
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel(r'$e_{1}$')

    ax1 = fig.add_subplot(637)
    ax1.plot(T, sol[:,6])
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel(r'$e_{2}$')

    ax1 = fig.add_subplot(638)
    ax1.plot(T, sol[:,7])
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel(r'$e_{3}$')

    ax1 = fig.add_subplot(639)
    ax1.plot(T, sol[:,8])
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel(r'$e_{4}$')

    ax1 = fig.add_subplot(6,3,10)
    ax1.plot(T, sol[:,9])
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel(r'$e_{5}$')

    ax1 = fig.add_subplot(6,3,11)
    ax1.plot(T, sol[:,10])
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel(r'$e_{6}$')

    ax1 = fig.add_subplot(6,3,12)
    ax1.plot(T, sol[:,11])
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel(r'$e_{7}$')

    ax1 = fig.add_subplot(6,3,13)
    ax1.plot(T, sol[:,12])
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel(r'$e_{8}$')

    ax1 = fig.add_subplot(6,3,14)
    ax1.plot(T, sol[:,13])
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel(r'$e_{9}$')

    ax1 = fig.add_subplot(6,3,15)
    ax1.plot(T, sol[:,14])
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel(r'$j_{v_{1}}$')

    ax1 = fig.add_subplot(6,3,16)
    ax1.plot(T, sol[:,15])
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel(r'$j_{v_{2}}$')

    ax1 = fig.add_subplot(6,3,17)
    ax1.plot(T, sol[:,16])
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel(r'$\lambda_{1}$')

    ax1 = fig.add_subplot(6,3,18)
    ax1.plot(T, sol[:,17])
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel(r'$\lambda_{2}$')

    plt.savefig(os.path.abspath(os.path.join(sacred_save_path,'../dc_microgrid_composite_ndae_trajectory.png')))
    plt.clf()

    return sol, composite_ndae

if __name__ == '__main__':
    x0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
    # y0 = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    y0 = jnp.array([100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    z0 = jnp.concatenate((x0, y0))
    T = jnp.linspace(0, 0.008, 800)
    # T = jnp.linspace(0, 800*1e-8, 800)
    # T = jnp.linspace(0, 1.5, 1000)
    exp_file_name = '2024-10-15_11-27-48_phdae_dgu_scalings'
    exp_file_name = '2024-09-20_12-24-41_train_phdae_dgu' # 1e-4
    dc_microgrid_ndae(z0, T, exp_file_name)