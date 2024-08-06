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

exp_file_name = '2024-08-05_17-23-01_train_phdae_rlc'
exp_file_name = '2024-08-06_11-18-03_train_phdae_rlc'
# exp_file_name = '2024-08-06_12-03-04_train_mlp_rlc'
sacred_save_path = os.path.abspath(os.path.join('../cyrus_experiments/runs/', exp_file_name, '1'))

config = load_config_file(sacred_save_path)
test_dataset_name = config['dataset_setup']['test_dataset_file_name']
test_dataset_file_path = config['dataset_setup']['dataset_path']
full_dataset_path = os.path.join(test_dataset_file_path, test_dataset_name)
with open(full_dataset_path, 'rb') as f:
    test_dataset = pickle.load(f)
model, params = load_model(sacred_save_path)
results = load_metrics(sacred_save_path)

predicted_trajectories = []
predicted_timesteps = []
for traj_ind in tqdm(range(test_dataset['state_trajectories'].shape[0])):
    init_state = test_dataset['state_trajectories'][traj_ind, 0, :]
    timesteps = test_dataset['timesteps'][traj_ind, :]
    traj_len = len(timesteps)
    predicted_traj, timesteps = predict_trajectory(model, params, init_state, traj_len, dt=0.01)
    predicted_trajectories.append(predicted_traj)
    predicted_timesteps.append(timesteps)
predict_trajectories = jnp.array(predicted_trajectories)
predicted_timesteps = jnp.array(predicted_timesteps)

# # Pull out a specific trajectory from the test dataset
# which_traj = 0
# timesteps = test_dataset['timesteps'][which_traj, :]
# traj_len = len(timesteps)
# initial_state = test_dataset['state_trajectories'][which_traj, 0, :]
# true_traj = test_dataset['state_trajectories'][which_traj, :, :]

# predicted_traj, timesteps = predict_trajectory(model, params, initial_state, traj_len)

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
# solved_traj = true_dae.solve(initial_state, timesteps, None)

# Plot the predicted trajectory
fontsize = 15

traj_to_plot = 0
predicted_traj = predicted_trajectories[traj_to_plot]
true_traj = test_dataset['state_trajectories'][traj_to_plot, :, :]

T = 0.01 * np.arange(0, traj_len)
fig = plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(321)
ax1.plot(T, predicted_traj[:,0], color='blue', linewidth=3, label='Predicted Dynamics')
ax1.plot(T, true_traj[:,0], color='black', linewidth=3, label='True Dynamics')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('q')

ax1 = fig.add_subplot(322)
ax1.plot(T, predicted_traj[:,1], color='blue', linewidth=3, label='Predicted Dynamics')
ax1.plot(T, true_traj[:,1], color='black', linewidth=3, label='True Dynamics')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('phi')

ax1 = fig.add_subplot(323)
ax1.plot(T, predicted_traj[:,2], color='blue', linewidth=3, label='Predicted Dynamics')
ax1.plot(T, true_traj[:,2], color='black', linewidth=3, label='True Dynamics')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('e1')

ax1 = fig.add_subplot(324)
ax1.plot(T, predicted_traj[:,3], color='blue', linewidth=3, label='Predicted Dynamics')
ax1.plot(T, true_traj[:,3], color='black', linewidth=3, label='True Dynamics')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('e2')

ax1 = fig.add_subplot(325)
ax1.plot(T, predicted_traj[:,4], color='blue', linewidth=3, label='Predicted Dynamics')
ax1.plot(T, true_traj[:,4], color='black', linewidth=3, label='True Dynamics')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('e3')

ax1 = fig.add_subplot(326)
ax1.plot(T, predicted_traj[:,5], color='blue', linewidth=3, label='Predicted Dynamics')
ax1.plot(T, true_traj[:,5], color='black', linewidth=3, label='True Dynamics')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

plt.savefig('phndae_predicted_trajectory.png')

# Now plot g values
g_vals_norm = []
g_vals = []
for traj_ind in range(predict_trajectories.shape[0]):
    predicted_traj = predict_trajectories[traj_ind]
    timesteps = predicted_timesteps[traj_ind]
    gnorm, gval = compute_g_vals_along_traj(true_dae.solver.g, params, predicted_traj, timesteps)
    g_vals_norm.append(gnorm)
    g_vals.append(gval)
g_vals_norm = jnp.array(g_vals_norm)
g_vals = jnp.array(g_vals)

median = jnp.median(g_vals, axis=0)
percentile_25 = jnp.percentile(g_vals, 25, axis=0)
percentile_75 = jnp.percentile(g_vals, 75, axis=0)

fig = plt.figure()
ax = fig.add_subplot(411)
ax.plot(median[:, 0])
ax.fill_between(range(len(median[:, 0])), percentile_25[:, 0], percentile_75[:, 0], alpha=0.5)

ax = fig.add_subplot(412)
ax.plot(median[:, 0])
ax.fill_between(range(len(median[:, 0])), percentile_25[:, 0], percentile_75[:, 0], alpha=0.5)

ax = fig.add_subplot(413)
ax.plot(median[:, 0])
ax.fill_between(range(len(median[:, 0])), percentile_25[:, 0], percentile_75[:, 0], alpha=0.5)

ax = fig.add_subplot(414)
ax.plot(median[:, 0])
ax.fill_between(range(len(median[:, 0])), percentile_25[:, 0], percentile_75[:, 0], alpha=0.5)

plt.savefig('g_for_true_rlc.png')

# Now for g vals squared
median = jnp.median(g_vals_norm, axis=0)
percentile_25 = jnp.percentile(g_vals_norm, 25, axis=0)
percentile_75 = jnp.percentile(g_vals_norm, 75, axis=0)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(median)
ax.fill_between(range(len(median)), percentile_25, percentile_75, alpha=0.5)
ax.set_xlabel('Time [s]')
ax.set_ylabel('||g||')

plt.savefig('g_norm_for_true_rlc.png')

# Now plot prediction errors.
traj_errs = []
for traj_ind in range(predict_trajectories.shape[0]):
    predicted_traj = predict_trajectories[traj_ind]
    true_traj = test_dataset['state_trajectories'][traj_ind, :, :]
    timesteps = predicted_timesteps[traj_ind]
    traj_err = compute_traj_err(true_traj, predicted_traj)
    traj_errs.append(traj_err)
traj_errs = jnp.array(traj_errs)

median = jnp.median(traj_errs, axis=0)
percentile_25 = jnp.percentile(traj_errs, 25, axis=0)
percentile_75 = jnp.percentile(traj_errs, 75, axis=0)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(median)
ax.fill_between(range(len(median)), percentile_25, percentile_75, alpha=0.5)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Trajectory Error')

plt.savefig('phndae_trajectory_error.png')