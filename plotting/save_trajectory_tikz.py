import os, sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from common import load_config_file, load_dataset, load_model, load_metrics, compute_traj_err, predict_trajectory, compute_g_vals_along_traj, tikzplotlib_fix_ncols, save_plot
import argparse
from models.ph_dae import PHDAE

import jax
# jax.default_device = jax.devices("gpu")[-1]

##### Experiment 1
exp_file_name = 'fhn/2024-11-12_18-27-53_fitz_hugh_nagano_less_data' # batch=128, pen_g=1e-2, lr=1e-4 cosine (same as dgu)

##### Experiment 2
exp_file_name = 'dgu_simple/2024-11-17_16-43-40_baseline' # batch=128, pen_g=1e-2, lr=1e-4 cosine
exp_file_name = 'batch_training/dgu_simple/0_baseline'

name = 'dgu'
tikz = True
fontsize = 15
traj_len = 499
phndae_color = (12/255, 212/255, 82/255)
mlp_color = (224/255, 95/255, 21/255)


sacred_save_path = os.path.abspath(os.path.join('../cyrus_experiments/runs/', exp_file_name, '1'))

config = load_config_file(sacred_save_path)

datasets = load_dataset(sacred_save_path)
results = load_metrics(sacred_save_path)

test_dataset = datasets['test_dataset']

model, params = load_model(sacred_save_path, config)

initial_state = test_dataset['inputs'][0, :-1]
true_traj = test_dataset['inputs'][0:(traj_len), :-1]
control_inputs = test_dataset['control_inputs'][0:(traj_len)]
predicted_traj, timesteps = predict_trajectory(model, params, initial_state, traj_len, config['model_setup']['dt'], control=control_inputs)

# Plot the predicted trajectory
T = model.dt * np.arange(0, traj_len)
fig = plt.figure(figsize=(10,4))

ax = fig.add_subplot(111)
for i in range(predicted_traj.shape[1]-1):
    ax.plot(T, true_traj[:,i], color='black', linewidth=3)
    ax.plot(T, predicted_traj[:,i], color=phndae_color, linewidth=3, ls='--')
ax.plot(T, true_traj[:,-1], color='black', linewidth=3, label='True Dynamics')
ax.plot(T, predicted_traj[:,-1], color=phndae_color, linewidth=3, ls='--', label='Predicted Dynamics')
ax.set_xlabel('Time')
plt.tight_layout()
save_plot(fig, tikz, name+"_predicted_trajectory")

# gnorm and traj err plot
gnorm, _ = compute_g_vals_along_traj(model.dae.g, params, predicted_traj, T, num_diff_vars=2, control=control_inputs)
err = compute_traj_err(true_traj, predicted_traj)
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111)
ax.plot(T, gnorm, color=phndae_color, linewidth=5)
ax.plot(T, err, color='black', linewidth=5)
ax.grid()
# ax.set_title(f'Subsystem constraint violation norm')
ax.set_xlabel('Time')
# ax.set_ylabel(r'$||g(x)||_2^2$')
ax.set_yscale('log')
save_plot(fig, tikz, name+"_err")
plt.clf()
plt.close()

# Plot g values
# gnorm, _ = compute_g_vals_along_traj(model.dae.g, params, predicted_traj, T_predicted, num_diff_vars=2, control=control_inputs)
# fig = plt.figure(figsize=(10, 4))
# ax = fig.add_subplot(111)
# ax.plot(gnorm, color=phndae_color, linewidth=5)
# ax.grid()
# ax.set_title(f'Subsystem constraint violation norm')
# ax.set_xlabel('Time')
# ax.set_ylabel(r'$||g(x)||_2^2$')
# save_plot(fig, tikz, name+"_g_vals")
# plt.clf()
# plt.close()

# err = compute_traj_err(true_traj, predicted_traj)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(err)
# ax.set_title(f'Subsystem trajectory error norm')
# ax.set_xlabel('Time')
# ax.set_ylabel(r'$||\hat{x} - x||_2^2$')
# save_plot(fig, tikz, name+"_traj_err")
# plt.clf()
# plt.close()

# print("Mean gnorm {:.2f}. Mean trajectory error {:.2f}".format(jnp.mean(gnorm), jnp.mean(err)))
