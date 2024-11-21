import os, sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import tikzplotlib

from common import load_config_file, load_dataset, load_model, load_metrics, compute_traj_err, predict_trajectory, compute_g_vals_along_traj, tikzplotlib_fix_ncols, save_plot
import argparse
from models.ph_dae import PHDAE

#### Experiment 1
# phndae_file_name = 'fhn/2024-11-12_18-27-53_fitz_hugh_nagano_less_data'
# phndae_file_name = 'fhn/2024-11-14_18-27-21_fitz_hugh_nagano_less_data' # batch=128, pen_g=1e-2, lr=1e-4 cosine (same as dgu) # NOTE
# node_file_name = 'fhn_node/2024-11-12_18-36-14_train_node_fhn'
# node_file_name = 'fhn_node/2024-11-12_21-31-38_train_node_fhn' # batch=256, pen_g=1e-1,lr=1e-3 cosine, less data
# node_file_name = 'fhn_node/2024-11-18_19-03-52_train_node_fhn' # more data NOTE

phndae_file_name = 'batch_training/fhn/0_baseline'
# node_file_name = 'fhn_node/2024-11-19_12-04-51_train_node_fhn' # lr=1e-3, batch=128 # more data
node_file_name = 'fhn_node/2024-11-19_12-59-57_train_node_fhn' # lr=1e-3, batch=128 # less data


from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--tikz', action='store_true')
args = parser.parse_args()

tikz = args.tikz

#### Takeaway, PHNDAE gives better accuracy and constraint satisfaction

phndae_save_path = os.path.abspath(os.path.join('../cyrus_experiments/runs/', phndae_file_name, '1'))
node_save_path = os.path.abspath(os.path.join('../cyrus_experiments/runs/', node_file_name, '1'))

phndae_color = (12/255, 212/255, 82/255)
node_color = (224/255, 95/255, 21/255)

phndae_config = load_config_file(phndae_save_path)
phndae_model, phndae_params = load_model(phndae_save_path)
node_config = load_config_file(node_save_path)
node_model, node_params = load_model(node_save_path)

datasets = load_dataset(phndae_save_path)
results = load_metrics(phndae_save_path)

test_dataset = datasets['test_dataset']

title = ""

traj_len = 1999
initial_state = test_dataset['inputs'][0, :-1]
true_traj = test_dataset['inputs'][0:(traj_len), :-1]
control_inputs = test_dataset['control_inputs'][0:(traj_len)]
phndae_traj, timesteps = predict_trajectory(phndae_model, phndae_params, initial_state, traj_len, phndae_config['model_setup']['dt'], control=control_inputs)
node_traj, timesteps = predict_trajectory(node_model, node_params, initial_state, traj_len, node_config['model_setup']['dt'], control=control_inputs)


# Plot the predicted trajectory
fontsize = 15

T = phndae_model.dt * np.arange(0, traj_len)

C = 1.0
L = 1 / 0.08

V_true = true_traj[:,0] / C
V_phndae = phndae_traj[:,0] / C
V_node = node_traj[:,0] / C

W_true = true_traj[:,1] / L
W_phndae = phndae_traj[:,1] / L
W_node = node_traj[:,1] / L

fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(111)
ax1.plot(T, V_true, color='black', linewidth=3, label='True')
ax1.plot(T, V_phndae, color=phndae_color, linewidth=3, ls='--', label='PHNDAE')
ax1.plot(T, V_node, color=node_color, linewidth=3, ls='--', label='NODE')
# ax1.plot(T, W_true, color='black', linewidth=3)
# ax1.plot(T, W_phndae, color=phndae_color, linewidth=3, ls='--')
# ax1.plot(T, W_node, color=node_color, linewidth=3, ls='--')
ax1.set_xlabel('Time [s]')
# ax1.set_ylim([-2.0, 2.0])
ax1.grid()
plt.tight_layout()
save_plot(fig, tikz, 'fhn_predicted_trajectory')
plt.clf()
plt.close()

# Plot g values
phndae_gnorm, _ = compute_g_vals_along_traj(phndae_model.dae.g, phndae_params, phndae_traj, T, num_diff_vars=2, control=control_inputs)
node_gnorm, _ = compute_g_vals_along_traj(phndae_model.dae.g, phndae_params, node_traj, T, num_diff_vars=2, control=control_inputs)
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
ax.plot(T, phndae_gnorm, color=phndae_color, linewidth=5)
ax.plot(T, node_gnorm, color=node_color, linewidth=5)
ax.grid()
# ax.set_title(f'Subsystem constraint violation norm {title}')
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'$||h(\omega)||_2^2$')
ax.set_yscale('log')
# ax.set_ylim([0.0,1e-2])
save_plot(fig, tikz, 'fhn_h_vals')
plt.clf()
plt.close()

phndae_err = compute_traj_err(true_traj, phndae_traj)
node_err = compute_traj_err(true_traj, node_traj)
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
ax.plot(T, phndae_err, color=phndae_color, linewidth=5)
ax.plot(T, node_err, color=node_color, linewidth=5)
# ax.set_ylim([0.0,1.0])
ax.grid()
ax.set_yscale('log')
# ax.set_title(f'Subsystem trajectory error norm {title}')
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'$||\hat{\omega} - \omega||_2^2$')
save_plot(fig, tikz, 'fhn_traj_err')
plt.clf()
plt.close()