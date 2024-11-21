import os, sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from common import load_config_file, load_dataset, load_model, load_metrics, compute_traj_err, predict_trajectory, compute_g_vals_along_traj
import argparse
from models.ph_dae import PHDAE

import jax
jax.default_device = jax.devices("gpu")[-1]

exp_file_name = '2024-08-05_17-23-01_train_phdae_rlc'
exp_file_name = '2024-08-06_12-03-04_train_mlp_rlc'
exp_file_name = '2024-09-19_11-57-38_train_phdae_dgu'
exp_file_name = '2024-09-20_14-54-03_train_phdae_dgu'
exp_file_name = '2024-09-23_19-45-01_train_phdae_dgu'
exp_file_name = '2024-10-22_12-25-00_phdae_dgu_user51' # user 5-1, pen_g=1e-2, 200000 epochs


exp_file_name = 'dgu/2024-11-11_09-30-54_phdae_dgu_user_1' # batch=256, pen_g=1e-1, lr=1e-6 NOTE this one is good...

# exp_file_name = 'dgu/2024-11-11_22-25-57_phdae_dgu_cosine' # batch=256, pen_g=1e-1, optax.schedules.cosine_decay_schedule(1e-5,5e5)
# exp_file_name = 'fhn/2024-11-11_22-25-28_fitz_hugh_nagano'
exp_file_name = 'fhn/2024-11-12_11-45-05_fitz_hugh_nagano_less_data'

exp_file_name = 'dgu/2024-11-12_12-46-59_phdae_dgu_cosine' # batch=64, pen_g=1e-1, lr=1e-5, less data, scaled voltage
exp_file_name = 'dgu/2024-11-12_13-09-24_phdae_dgu_cosine' # batch=512, pen_g=1e-1, lr=1e-5, less data, scaled voltage
exp_file_name = 'dgu/2024-11-12_13-36-59_phdae_dgu_less_data' # batch=128, pen_g=1e-1, lr=1e-4
exp_file_name = 'dgu/2024-11-12_17-00-11_phdae_dgu_less_data' # batch=128, pen_g=1e-2, lr=1e-4 cosine

#### Experiment 1
exp_file_name = 'fhn/2024-11-12_18-27-53_fitz_hugh_nagano_less_data' # batch=128, pen_g=1e-2, lr=1e-4 cosine (same as dgu)
# exp_file_name = 'fhn_node/2024-11-12_18-36-14_train_node_fhn' # lr=1e-4 cosine, 100000, batch=128
#### Takeaway, PHNDAE gives better accuracy and constraint satisfaction

# exp_file_name = 'dgu/2024-11-13_21-09-12_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-4 cosine scale=50 # WORKS!
exp_file_name = 'dgu/2024-11-14_20-21-20_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-4 cosine, 1e5 epochs, new data, scale=5, 3e5
exp_file_name = 'dgu/2024-11-15_17-58-15_phdae_dgu'  # all one component values, lr=1e-3
exp_file_name = 'dgu_simple/2024-11-16_16-30-35_more_batch128_lr1e-3'
exp_file_name = 'fhn_node/2024-11-18_19-03-52_train_node_fhn' # more data

##### Experiment 2
# exp_file_name = 'dgu_simple/2024-11-17_16-43-40_baseline' # batch=128, pen_g=1e-2, lr=1e-4 cosine


sacred_save_path = os.path.abspath(os.path.join('../cyrus_experiments/runs/', exp_file_name, '1'))

phndae_color = (12/255, 212/255, 82/255)
mlp_color = (224/255, 95/255, 21/255)

config = load_config_file(sacred_save_path)

datasets = load_dataset(sacred_save_path)
results = load_metrics(sacred_save_path)

test_dataset = datasets['test_dataset']

model, params = load_model(sacred_save_path, config)

title = ""

traj_len = 499
initial_state = test_dataset['inputs'][0, :-1]
true_traj = test_dataset['inputs'][0:(traj_len), :-1]
control_inputs = test_dataset['control_inputs'][0:(traj_len)]
predicted_traj, timesteps = predict_trajectory(model, params, initial_state, traj_len, config['model_setup']['dt'], control=control_inputs)

# Plot the predicted trajectory
fontsize = 15

T_true = model.dt * np.arange(0, traj_len)
T_predicted = T_true
fig = plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(321)
ax1.plot(T_true, true_traj[:,0], color='black', linewidth=3, label='True Dynamics')
ax1.plot(T_predicted, predicted_traj[:,0], color=phndae_color, linewidth=3, ls='--', label='Predicted Dynamics')
ax1.set_ylim([-2.0,2.0])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$q$')

ax1 = fig.add_subplot(322)
ax1.plot(T_true, true_traj[:,1], color='black', linewidth=3, label='True Dynamics')
ax1.plot(T_predicted, predicted_traj[:,1], color=phndae_color, linewidth=3, ls='--', label='Predicted Dynamics')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$\phi$')

ax1 = fig.add_subplot(323)
ax1.plot(T_true, true_traj[:,2], color='black', linewidth=3, label='True Dynamics')
ax1.plot(T_predicted, predicted_traj[:,2], color=phndae_color, linewidth=3, ls='--', label='Predicted Dynamics')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$e_1$')

ax1 = fig.add_subplot(324)
ax1.plot(T_true, true_traj[:,3], color='black', linewidth=3, label='True Dynamics')
ax1.plot(T_predicted, predicted_traj[:,3], color=phndae_color, linewidth=3, ls='--', label='Predicted Dynamics')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$e_2$')

ax1 = fig.add_subplot(325)
ax1.plot(T_true, true_traj[:,4], color='black', linewidth=3, label='True Dynamics')
ax1.plot(T_predicted, predicted_traj[:,4], color=phndae_color, linewidth=3, ls='--', label='Predicted Dynamics')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$e_3$')

ax1 = fig.add_subplot(326)
ax1.plot(T_true, true_traj[:,5], color='black', linewidth=3, label='True Dynamics')
ax1.plot(T_predicted, predicted_traj[:,5], color=phndae_color, linewidth=3, ls='--', label='Predicted Dynamics')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$j_v$')
plt.tight_layout()
plt.savefig('phndae_predicted_trajectory.png')

# Plot g values
gnorm, _ = compute_g_vals_along_traj(model.dae.g, params, predicted_traj, T_predicted, num_diff_vars=2, control=control_inputs)
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111)
ax.plot(gnorm, color=phndae_color, linewidth=5)
ax.grid()
ax.set_title(f'Subsystem constraint violation norm {title}')
ax.set_xlabel('Time')
ax.set_ylabel(r'$||g(x)||_2^2$')
plt.savefig(f'mlp_g_vals.png', dpi=600)
plt.clf()
plt.close()

err = compute_traj_err(true_traj, predicted_traj)
# err = compute_traj_err(true_traj[::8][:100], predicted_traj)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(err)
ax.set_title(f'Subsystem trajectory error norm {title}')
ax.set_xlabel('Time')
ax.set_ylabel(r'$||\hat{x} - x||_2^2$')
plt.savefig(f'mlp_traj_err.png', dpi=600)
plt.clf()
plt.close()

print("Mean gnorm {:.2f}. Mean trajectory error {:.2f}".format(jnp.mean(gnorm), jnp.mean(err)))

# Plot the violation of the true model constraints
# Build the true system.
# AC = jnp.array([[0.0], [0.0], [1.0]])
# AR = jnp.array([[1.0], [-1.0], [0.0]])
# AL = jnp.array([[0.0], [1.0], [-1.0]])
# AV = jnp.array([[1.0], [0.0], [0.0]])
# AI = jnp.array([[0.0], [0.0], [0.0]])

# R = 1
# L = 1
# C = 1

# def r_func(delta_V, params=None):
#     return delta_V / R

# def q_func(delta_V, params=None):
#     return C * delta_V

# def grad_H_func(phi, params=None):
#     return phi / L

# def u_func(t, params):
#     return jnp.array([jnp.sin(30 * t)])

# true_dae = PHDAE(AC, AR, AL, AV, AI, grad_H_func=grad_H_func, q_func=q_func, r_func=r_func, u_func=u_func)

# # true_traj = true_dae.solve(z0, T, None)
# # predicted_traj = model.predict_trajectory(params, initial_state=z0, num_steps=traj_len)

# g_vals = []
# for t_ind in range(predicted_traj.shape[0]):
#     t = T[t_ind]
#     z = predicted_traj[t_ind, :]
#     x = z[0:2]
#     y = z[2::]
#     g_vals.append(true_dae.solver.g(x,y,t,params))

# g_vals = jnp.array(g_vals)
# fig = plt.figure()
# ax = fig.add_subplot(411)
# ax.plot(g_vals[:, 0])

# ax = fig.add_subplot(412)
# ax.plot(g_vals[:, 1])

# ax = fig.add_subplot(413)
# ax.plot(g_vals[:, 2])

# ax = fig.add_subplot(414)
# ax.plot(g_vals[:, 3])

# plt.savefig('g_for_true_rlc.png')