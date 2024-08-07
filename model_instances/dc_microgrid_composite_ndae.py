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

ph_dae_list = []
params_list = []

# Load the DGU1 model
exp_file_name = '2024-08-06_18-40-56_train_phdae_dgu'
sacred_save_path = os.path.abspath(os.path.join('../cyrus_experiments/runs/', exp_file_name, '1'))

config = load_config_file(sacred_save_path)

config = load_config_file(sacred_save_path)
model_setup = config['model_setup']
model_setup['u_func_freq'] = 0.0
model_setup['u_func_current_source_magnitude'] = 0.1
model_setup['u_func_voltage_source_magnitude'] = 1.0
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

R = 1
L = 1

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
exp_file_name = '2024-08-06_18-40-56_train_phdae_dgu'
sacred_save_path = os.path.abspath(os.path.join('../cyrus_experiments/runs/', exp_file_name, '1'))

config = load_config_file(sacred_save_path)

config = load_config_file(sacred_save_path)
model_setup = config['model_setup']
model_setup['u_func_freq'] = 0.0
model_setup['u_func_current_source_magnitude'] = 0.1
model_setup['u_func_voltage_source_magnitude'] = 1.0
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
composite_ndae = CompositePHDAE(ph_dae_list, A_lambda)

x0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
y0 = jnp.array([9.99999992e-01, 9.99999992e-01, -1.99715686e-38, -3.99431372e-38, -4.18052975e-29,  0.00000000e+00,  9.99999992e-01,  9.99999992e-01,
  0.00000000e+00,  9.90870208e-29,  7.03697750e-29,  8.36221411e-29, 1.03092021e-29])
z0 = jnp.concatenate((x0, y0))
T = jnp.linspace(0, 1.5, 1000)

sol = composite_ndae.solve(z0, T, params_list=params_list)

print(sol.shape)

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

plt.savefig('dc_microgrid_composite_ndae_trajectory.png')