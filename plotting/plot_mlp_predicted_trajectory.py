import os, sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from common import load_config_file, load_dataset, load_model, load_metrics, compute_traj_err, predict_trajectory
import argparse
from models.ph_dae import PHDAE

exp_file_name = '2024-08-05_17-23-01_train_phdae_rlc'
exp_file_name = '2024-08-06_12-03-04_train_mlp_rlc'
sacred_save_path = os.path.abspath(os.path.join('../cyrus_experiments/runs/', exp_file_name, '1'))

config = load_config_file(sacred_save_path)
model, params = load_model(sacred_save_path)
datasets = load_dataset(sacred_save_path)
results = load_metrics(sacred_save_path)

test_dataset = datasets['test_dataset']

traj_len = 99
initial_state = test_dataset['inputs'][0, :-1]
true_traj = test_dataset['inputs'][0:(traj_len), :-1]
# predicted_traj = model.predict_trajectory(params, initial_state, traj_len)['state_trajectory']
predicted_traj, timesteps = predict_trajectory(model, params, initial_state, traj_len)

# Plot the predicted trajectory
fontsize = 15

T = model.dt * np.arange(0, traj_len)
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

# Plot the violation of the true model constraints
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

# true_traj = true_dae.solve(z0, T, None)
# predicted_traj = model.predict_trajectory(params, initial_state=z0, num_steps=traj_len)

g_vals = []
for t_ind in range(predicted_traj.shape[0]):
    t = T[t_ind]
    z = predicted_traj[t_ind, :]
    x = z[0:2]
    y = z[2::]
    g_vals.append(true_dae.solver.g(x,y,t,params))

g_vals = jnp.array(g_vals)
fig = plt.figure()
ax = fig.add_subplot(411)
ax.plot(g_vals[:, 0])

ax = fig.add_subplot(412)
ax.plot(g_vals[:, 1])

ax = fig.add_subplot(413)
ax.plot(g_vals[:, 2])

ax = fig.add_subplot(414)
ax.plot(g_vals[:, 3])

plt.savefig('g_for_true_rlc.png')