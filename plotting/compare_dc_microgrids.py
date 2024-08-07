import sys, os
sys.path.append('../')
import jax.numpy as jnp

from model_instances.dc_microgrid_composite_dae import composite_dae
from model_instances.dc_microgrid_composite_ndae import composite_ndae, params_list
from common import compute_g_vals_along_traj, compute_traj_err

import matplotlib.pyplot as plt

x0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
y0 = jnp.array([9.99999992e-01, 9.99999992e-01, -1.99715686e-38, -3.99431372e-38, -4.18052975e-29,  0.00000000e+00,  9.99999992e-01,  9.99999992e-01,
  0.00000000e+00,  9.90870208e-29,  7.03697750e-29,  8.36221411e-29, 1.03092021e-29])
z0 = jnp.concatenate((x0, y0))
T = jnp.linspace(0, 1.5, 1000)

dae_sol = composite_dae.solve(z0, T, params_list=[None, None, None])
ndae_sol = composite_ndae.solve(z0, T, params_list=params_list)


fig = plt.figure(figsize=(10, 20))

ax1 = fig.add_subplot(631)
ax1.plot(T, dae_sol[:,0], linewidth=2, color='black')
ax1.plot(T, ndae_sol[:,0], linewidth=2, color='blue', linestyle='--')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$q_{1}$')

ax1 = fig.add_subplot(632)
ax1.plot(T, dae_sol[:,1], linewidth=2, color='black')
ax1.plot(T, ndae_sol[:,1], linewidth=2, color='blue', linestyle='--')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$q_{2}$')

ax1 = fig.add_subplot(633)
ax1.plot(T, dae_sol[:,2], linewidth=2, color='black')
ax1.plot(T, ndae_sol[:,2], linewidth=2, color='blue', linestyle='--')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$\phi_{1}$')

ax1 = fig.add_subplot(634)
ax1.plot(T, dae_sol[:,3], linewidth=2, color='black')
ax1.plot(T, ndae_sol[:,3], linewidth=2, color='blue', linestyle='--')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$\phi_{2}$')

ax1 = fig.add_subplot(635)
ax1.plot(T, dae_sol[:,4], linewidth=2, color='black')
ax1.plot(T, ndae_sol[:,4], linewidth=2, color='blue', linestyle='--')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$\phi_{3}$')

ax1 = fig.add_subplot(636)
ax1.plot(T, dae_sol[:,5], linewidth=2, color='black')
ax1.plot(T, ndae_sol[:,5], linewidth=2, color='blue', linestyle='--')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$e_{1}$')

ax1 = fig.add_subplot(637)
ax1.plot(T, dae_sol[:,6], linewidth=2, color='black')
ax1.plot(T, ndae_sol[:,6], linewidth=2, color='blue', linestyle='--')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$e_{2}$')

ax1 = fig.add_subplot(638)
ax1.plot(T, dae_sol[:,7], linewidth=2, color='black')
ax1.plot(T, ndae_sol[:,7], linewidth=2, color='blue', linestyle='--')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$e_{3}$')

ax1 = fig.add_subplot(639)
ax1.plot(T, dae_sol[:,8], linewidth=2, color='black')
ax1.plot(T, ndae_sol[:,8], linewidth=2, color='blue', linestyle='--')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$e_{4}$')

ax1 = fig.add_subplot(6,3,10)
ax1.plot(T, dae_sol[:,9], linewidth=2, color='black')
ax1.plot(T, ndae_sol[:,9], linewidth=2, color='blue', linestyle='--')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$e_{5}$')

ax1 = fig.add_subplot(6,3,11)
ax1.plot(T, dae_sol[:,10], linewidth=2, color='black')
ax1.plot(T, ndae_sol[:,10], linewidth=2, color='blue', linestyle='--')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$e_{6}$')

ax1 = fig.add_subplot(6,3,12)
ax1.plot(T, dae_sol[:,11], linewidth=2, color='black')
ax1.plot(T, ndae_sol[:,11], linewidth=2, color='blue', linestyle='--')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$e_{7}$')

ax1 = fig.add_subplot(6,3,13)
ax1.plot(T, dae_sol[:,12], linewidth=2, color='black')
ax1.plot(T, ndae_sol[:,12], linewidth=2, color='blue', linestyle='--')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$e_{8}$')

ax1 = fig.add_subplot(6,3,14)
ax1.plot(T, dae_sol[:,13], linewidth=2, color='black')
ax1.plot(T, ndae_sol[:,13], linewidth=2, color='blue', linestyle='--')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$e_{9}$')

ax1 = fig.add_subplot(6,3,15)
ax1.plot(T, dae_sol[:,14], linewidth=2, color='black')
ax1.plot(T, ndae_sol[:,14], linewidth=2, color='blue', linestyle='--')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$j_{v_{1}}$')

ax1 = fig.add_subplot(6,3,16)
ax1.plot(T, dae_sol[:,15], linewidth=2, color='black')
ax1.plot(T, ndae_sol[:,15], linewidth=2, color='blue', linestyle='--')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$j_{v_{2}}$')

ax1 = fig.add_subplot(6,3,17)
ax1.plot(T, dae_sol[:,16], linewidth=2, color='black')
ax1.plot(T, ndae_sol[:,16], linewidth=2, color='blue', linestyle='--')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$\lambda_{1}$')

ax1 = fig.add_subplot(6,3,18)
ax1.plot(T, dae_sol[:,17], linewidth=2, color='black')
ax1.plot(T, ndae_sol[:,17], linewidth=2, color='blue', linestyle='--')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$\lambda_{2}$')

plt.savefig('comparison_dc_microgrid_trajectories.png')


# Plot g values
gnorm, _ = compute_g_vals_along_traj(composite_dae.solver.g, [None, None, None], dae_sol, T, num_diff_vars=5)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(gnorm)
plt.savefig('comparison_dc_microgrid_g_vals.png')

# Compute trajectory error
err = compute_traj_err(dae_sol, ndae_sol)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(err)
plt.savefig('comparison_dc_microgrid_traj_err.png')