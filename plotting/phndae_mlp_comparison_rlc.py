import os, sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from common import load_config_file, load_dataset, load_model, load_metrics, predict_trajectory, compute_traj_err, compute_g_vals_along_traj
import argparse
from models.ph_dae import PHDAE
import pickle
from tqdm import tqdm

phndae_color = (12/255, 212/255, 82/255)
mlp_color = (224/255, 95/255, 21/255)

mlp_exp_file_name = '2024-08-06_12-03-04_train_mlp_rlc'
phndae_exp_file_name = '2024-08-06_16-41-37_train_phdae_rlc'
mlp_sacred_save_path = os.path.abspath(os.path.join('../cyrus_experiments/runs/', mlp_exp_file_name, '1'))
phndae_sacred_save_path = os.path.abspath(os.path.join('../cyrus_experiments/runs/', phndae_exp_file_name, '1'))

mlp_config = load_config_file(mlp_sacred_save_path)
test_dataset_name = mlp_config['dataset_setup']['test_dataset_file_name']
test_dataset_file_path = mlp_config['dataset_setup']['dataset_path']

full_dataset_path = os.path.join(test_dataset_file_path, test_dataset_name)
with open(full_dataset_path, 'rb') as f:
    test_dataset = pickle.load(f)

mlp_model, mlp_params = load_model(mlp_sacred_save_path)
mlp_results = load_metrics(mlp_sacred_save_path)

phndae_model, ph_ndae_params = load_model(phndae_sacred_save_path)
phndae_results = load_metrics(phndae_sacred_save_path)

mlp_predicted_trajectories = []
mlp_predicted_timesteps = []

phndae_predicted_trajectories = []
phndae_predicted_timesteps = []

for traj_ind in tqdm(range(test_dataset['state_trajectories'].shape[0])):
    init_state = test_dataset['state_trajectories'][traj_ind, 0, :]
    timesteps = test_dataset['timesteps'][traj_ind, :]
    traj_len = len(timesteps)

    mlp_predicted_traj, mlp_timesteps = predict_trajectory(mlp_model, mlp_params, init_state, traj_len, dt=0.01)
    mlp_predicted_trajectories.append(mlp_predicted_traj)
    mlp_predicted_timesteps.append(mlp_timesteps)

    phndae_predicted_traj, phndae_timesteps = predict_trajectory(phndae_model, ph_ndae_params, init_state, traj_len, dt=0.01)
    phndae_predicted_trajectories.append(phndae_predicted_traj)
    phndae_predicted_timesteps.append(phndae_timesteps)

mlp_predicted_trajectories = jnp.array(mlp_predicted_trajectories)
mlp_predicted_timesteps = jnp.array(mlp_predicted_timesteps)

phndae_predicted_trajectories = jnp.array(phndae_predicted_trajectories)
phndae_predicted_timesteps = jnp.array(phndae_predicted_timesteps)

# Build the true system.
AC = jnp.array([[0.0], [0.0], [1.0]])
AR = jnp.array([[1.0], [-1.0], [0.0]])
AL = jnp.array([[0.0], [1.0], [-1.0]])
AV = jnp.array([[1.0], [0.0], [0.0]])
AI = jnp.array([[0.0], [0.0], [0.0]])

R = 1
L = 1
C = 1

def r_func(delta_V, params=None):
    return delta_V / R

def q_func(delta_V, params=None):
    return C * delta_V

def grad_H_func(phi, params=None):
    return phi / L

def u_func(t, params):
    return jnp.array([jnp.sin(30 * t)])

true_dae = PHDAE(AC, AR, AL, AV, AI, grad_H_func=grad_H_func, q_func=q_func, r_func=r_func, u_func=u_func)

# Plot the predicted trajectory
fontsize = 15

traj_to_plot = 0
mlp_predicted_traj = mlp_predicted_trajectories[traj_to_plot]
phndae_predicted_traj = phndae_predicted_trajectories[traj_to_plot]
true_traj = test_dataset['state_trajectories'][traj_to_plot, :, :]

T = 0.01 * np.arange(0, traj_len)
fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(321)
ax1.plot(T, true_traj[:,0], color='black', linewidth=8)
ax1.plot(T, mlp_predicted_traj[:,0], color=mlp_color, linewidth=5, linestyle='--')
ax1.plot(T, phndae_predicted_traj[:,0], color=phndae_color, linewidth=5, linestyle='--')
ax1.grid()

ax1 = fig.add_subplot(322)
ax1.plot(T, true_traj[:,1], color='black', linewidth=8)
ax1.plot(T, mlp_predicted_traj[:,1], color=mlp_color, linewidth=5, linestyle='--')
ax1.plot(T, phndae_predicted_traj[:,1], color=phndae_color, linewidth=5, linestyle='--')
ax1.grid()

ax1 = fig.add_subplot(323)
ax1.plot(T, true_traj[:,2], color='black', linewidth=8)
ax1.plot(T, mlp_predicted_traj[:,2], color=mlp_color, linewidth=5, linestyle='--')
ax1.plot(T, phndae_predicted_traj[:,2], color=phndae_color, linewidth=5, linestyle='--')
ax1.grid()

ax1 = fig.add_subplot(324)
ax1.plot(T, true_traj[:,3], color='black', linewidth=8)
ax1.plot(T, mlp_predicted_traj[:,3], color=mlp_color, linewidth=5, linestyle='--')
ax1.plot(T, phndae_predicted_traj[:,3], color=phndae_color, linewidth=5, linestyle='--')
ax1.grid()

ax1 = fig.add_subplot(325)
ax1.plot(T, true_traj[:,4], color='black', linewidth=8)
ax1.plot(T, mlp_predicted_traj[:,4], color=mlp_color, linewidth=5, linestyle='--')
ax1.plot(T, phndae_predicted_traj[:,4], color=phndae_color, linewidth=5, linestyle='--')
ax1.grid()

ax1 = fig.add_subplot(326)
ax1.plot(T, true_traj[:,5], color='black', linewidth=5)
ax1.plot(T, mlp_predicted_traj[:,5], color=mlp_color, linewidth=3, linestyle='--')
ax1.plot(T, phndae_predicted_traj[:,5], color=phndae_color, linewidth=3, linestyle='--')
ax1.grid()

plt.savefig('compare_predicted_trajectories_rlc.png', dpi=600)
# tikzplotlib.save('compare_predicted_trajectories_rlc.tex')

# Now plot g values
mlp_g_vals_norm = []
phndae_g_vals_norm = []
for traj_ind in range(mlp_predicted_trajectories.shape[0]):
    mlp_predicted_traj = mlp_predicted_trajectories[traj_ind]
    mlp_timesteps = mlp_predicted_timesteps[traj_ind]
    gnorm, _ = compute_g_vals_along_traj(true_dae.solver.g, mlp_params, mlp_predicted_traj, mlp_timesteps, num_diff_vars=2)
    mlp_g_vals_norm.append(gnorm)

    phndae_predicted_traj = phndae_predicted_trajectories[traj_ind]
    phndae_timesteps = phndae_predicted_timesteps[traj_ind]
    gnorm, _ = compute_g_vals_along_traj(true_dae.solver.g, ph_ndae_params, phndae_predicted_traj, phndae_timesteps, num_diff_vars=2)
    phndae_g_vals_norm.append(gnorm)

mlp_g_vals_norm = jnp.array(mlp_g_vals_norm)
phndae_g_vals_norm = jnp.array(phndae_g_vals_norm)

mlp_median = jnp.median(mlp_g_vals_norm, axis=0)
mlp_percentile_25 = jnp.percentile(mlp_g_vals_norm, 25, axis=0)
mlp_percentile_75 = jnp.percentile(mlp_g_vals_norm, 75, axis=0)

phndae_median = jnp.median(phndae_g_vals_norm, axis=0)
phndae_percentile_25 = jnp.percentile(phndae_g_vals_norm, 25, axis=0)
phndae_percentile_75 = jnp.percentile(phndae_g_vals_norm, 75, axis=0)

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
ax.fill_between(range(len(mlp_median)), mlp_percentile_25, mlp_percentile_75, alpha=0.3, color=mlp_color)
ax.plot(mlp_median, linewidth=3, color=mlp_color)

ax.fill_between(range(len(mlp_median)), phndae_percentile_25, phndae_percentile_75, alpha=0.3, color=phndae_color)
ax.plot(phndae_median, linewidth=3, color=phndae_color)
ax.grid()

plt.savefig('compare_g_norm_for_rlc.png', dpi=600)

# Now plot prediction errors.
mlp_traj_errs = []
phndae_traj_errs = []
for traj_ind in range(mlp_predicted_trajectories.shape[0]):
    true_traj = test_dataset['state_trajectories'][traj_ind, :, :]

    mlp_predicted_traj = mlp_predicted_trajectories[traj_ind]
    timesteps = mlp_predicted_timesteps[traj_ind]
    mlp_traj_err = compute_traj_err(true_traj, mlp_predicted_traj)
    mlp_traj_errs.append(mlp_traj_err)

    phndae_predicted_traj = phndae_predicted_trajectories[traj_ind]
    timesteps = phndae_predicted_timesteps[traj_ind]
    phndae_traj_err = compute_traj_err(true_traj, phndae_predicted_traj)
    phndae_traj_errs.append(phndae_traj_err)

mlp_traj_errs = jnp.array(mlp_traj_errs)
phndae_traj_errs = jnp.array(phndae_traj_errs)

mlp_median = jnp.median(mlp_traj_errs, axis=0)
mlp_percentile_25 = jnp.percentile(mlp_traj_errs, 25, axis=0)
mlp_percentile_75 = jnp.percentile(mlp_traj_errs, 75, axis=0)

phndae_median = jnp.median(phndae_traj_errs, axis=0)
phndae_percentile_25 = jnp.percentile(phndae_traj_errs, 25, axis=0)
phndae_percentile_75 = jnp.percentile(phndae_traj_errs, 75, axis=0)

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
ax.fill_between(range(len(mlp_median)), mlp_percentile_25, mlp_percentile_75, color=mlp_color, alpha=0.3)
ax.plot(mlp_median, linewidth=3, color=mlp_color)

ax.fill_between(range(len(phndae_median)), phndae_percentile_25, phndae_percentile_75, alpha=0.3, color=phndae_color)
ax.plot(phndae_median, linewidth=3, color=phndae_color)
ax.grid()

plt.savefig('compare_trajectory_error_rlc.png', dpi=600)