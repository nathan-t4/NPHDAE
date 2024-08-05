import os, sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from common import load_config_file, load_dataset, load_model, load_metrics
import argparse

exp_file_name = '2024-08-04_23-12-37_train_phdae_rlc'
sacred_save_path = os.path.abspath(os.path.join('../cyrus_experiments/runs/', exp_file_name, '1'))

config = load_config_file(sacred_save_path)
model, params = load_model(sacred_save_path)
datasets = load_dataset(sacred_save_path)
results = load_metrics(sacred_save_path)

test_dataset = datasets['test_dataset']

traj_len = 500
initial_state = test_dataset['outputs'][0, :]
true_traj = test_dataset['outputs'][0:traj_len, :]
predicted_traj_and_control = model.predict_trajectory(params, initial_state=initial_state, 
                                            num_steps=traj_len)
predicted_traj = predicted_traj_and_control['state_trajectory']

# Generate a predicted trajectory
fontsize = 15

T = model.dt * np.arange(0, traj_len)
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(211)
ax.plot(T, predicted_traj[:,0], color='blue', linewidth=3, label='Predicted Dynamics')
ax.plot(T, true_traj[:,0], color='black', linewidth=3, label='True Dynamics')
ax.legend(fontsize=fontsize)
ax.set_xlabel('Time [s]', fontsize=fontsize)
ax.set_ylabel(r'$x$ $[m]$', fontsize=fontsize)

ax = fig.add_subplot(212)
ax.plot(T, predicted_traj[:,1], color='blue', linewidth=3, label='Predicted Dynamics')
ax.plot(T, true_traj[:,1], color='black', linewidth=3, label='True Dynamics')
ax.set_xlabel('Time [s]', fontsize=fontsize)
ax.set_ylabel(r'$\frac{dx}{dt}$ $[\frac{m}{s}]$', fontsize=fontsize)

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(T, predicted_control[:,0], color='blue', linewidth=3, label='Predicted Control')
plt.show()