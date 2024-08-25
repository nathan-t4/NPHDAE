import sys
sys.path.append('../')
from models.ph_dae import PHDAE
from models.composite_ph_dae import CompositePHDAE
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

ph_dae_list = []

# DGU 1
AC = jnp.array([[0.0], [0.0], [1.0]])
AR = jnp.array([[-1.0], [1.0], [0.0]])
AL = jnp.array([[0.0], [1.0], [-1.0]])
AV = jnp.array([[1.0], [0.0], [0.0]])
AI = jnp.array([[0.0], [0.0], [-1.0]])

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
    return jnp.array([0.1, 1.0])

dgu1 = PHDAE(AC, AR, AL, AV, AI, grad_H_func, q_func, r_func, u_func)
ph_dae_list.append(dgu1)

# DGU 2
AC = jnp.array([[0.0], [0.0], [1.0]])
AR = jnp.array([[-1.0], [1.0], [0.0]])
AL = jnp.array([[0.0], [1.0], [-1.0]])
AV = jnp.array([[1.0], [0.0], [0.0]])
AI = jnp.array([[0.0], [0.0], [-1.0]])

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
    return jnp.array([0.1, 1.0])

dgu2 = PHDAE(AC, AR, AL, AV, AI, grad_H_func, q_func, r_func, u_func)
ph_dae_list.append(dgu2)

# DGU 3
AC = jnp.array([[0.0], [0.0], [1.0]])
AR = jnp.array([[-1.0], [1.0], [0.0]])
AL = jnp.array([[0.0], [1.0], [-1.0]])
AV = jnp.array([[1.0], [0.0], [0.0]])
AI = jnp.array([[0.0], [0.0], [-1.0]])

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
    return jnp.array([0.1, 1.0])

dgu3 = PHDAE(AC, AR, AL, AV, AI, grad_H_func, q_func, r_func, u_func)
ph_dae_list.append(dgu3)

# Transmission line 1
AC = jnp.array([[0.0], [0.0], [0.0]])
AR = jnp.array([[1.0], [-1.0], [0.0]])
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

transmission_line_1 = PHDAE(AC, AR, AL, AV, AI, grad_H_func, q_func, r_func, u_func)
ph_dae_list.append(transmission_line_1)

# Transmission line 2
AC = jnp.array([[0.0], [0.0], [0.0]])
AR = jnp.array([[1.0], [-1.0], [0.0]])
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

transmission_line_2 = PHDAE(AC, AR, AL, AV, AI, grad_H_func, q_func, r_func, u_func)
ph_dae_list.append(transmission_line_2)

# Transmission line 3
AC = jnp.array([[0.0], [0.0], [0.0]])
AR = jnp.array([[1.0], [-1.0], [0.0]])
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

transmission_line_3 = PHDAE(AC, AR, AL, AV, AI, grad_H_func, q_func, r_func, u_func)
ph_dae_list.append(transmission_line_3)

A_lambda = np.zeros((18,6))
A_lambda[2,0] = 1; A_lambda[9,0] = -1
A_lambda[5,1] = 1; A_lambda[11,1] = -1
A_lambda[2,2] = 1; A_lambda[12,2] = -1
A_lambda[8,3] = 1; A_lambda[14,3] = -1
A_lambda[5,4] = 1; A_lambda[15,4] = -1
A_lambda[8,5] = 1; A_lambda[17,5] = -1
A_lambda = jnp.array(A_lambda)

composite_dae = CompositePHDAE(ph_dae_list, A_lambda)

x0 = jnp.zeros(9)
y0 = jnp.zeros(27) # num_nodes+num_volt_sources+num_couplings
z0 = jnp.concatenate((x0, y0))
T = jnp.linspace(0, 1.5, 1000)

sol = composite_dae.solve(z0, T, params_list=[None] * len(ph_dae_list))

print(sol.shape)

from plotting.common import compute_g_vals_along_traj
gnorm, gval = compute_g_vals_along_traj(composite_dae.solver.g, [None] * len(ph_dae_list), sol, T, num_diff_vars=9)

phndae_color = (12/255, 212/255, 82/255)
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
ax.plot(T, gnorm*1000, color=phndae_color, linewidth=5)

plt.savefig('dgu_triangle_gnorm_new.png')

from environments.dgu_triangle_dae import generate_dataset
exp_traj = generate_dataset(ntimesteps=1500, plot=False)['state_trajectories'][0]

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
ax.plot(T, exp_traj[:,14], color='black', linewidth=5)
ax.plot(T, sol[:,14], '--', color=phndae_color, linewidth=5)
plt.savefig('dgu_triangle_rollout_new.png')

fig = plt.figure(figsize=(10, 20))

ax1 = fig.add_subplot(511)
ax1.plot(T, sol[:,0], label=r'$q_{1}$')
ax1.plot(T, sol[:,1], label=r'$q_{2}$')
ax1.plot(T, sol[:,2], label=r'$q_{3}$')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$q$')
ax1.legend()

ax1 = fig.add_subplot(512)
ax1.plot(T, sol[:,3], label=r'$\phi_{1}$')
ax1.plot(T, sol[:,4], label=r'$\phi_{2}$')
ax1.plot(T, sol[:,5], label=r'$\phi_{3}$')
ax1.plot(T, sol[:,6], label=r'$\phi_{4}$')
ax1.plot(T, sol[:,7], label=r'$\phi_{5}$')
ax1.plot(T, sol[:,8], label=r'$\phi_{6}$')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$\phi$')
ax1.legend()


ax1 = fig.add_subplot(513)
ax1.plot(T, sol[:,9], label=r'$e_{1}$')
ax1.plot(T, sol[:,10], label=r'$e_{2}$')
ax1.plot(T, sol[:,11], label=r'$e_{3}$')
ax1.plot(T, sol[:,12], label=r'$e_{4}$')
ax1.plot(T, sol[:,13], label=r'$e_{5}$')
ax1.plot(T, sol[:,14], label=r'$e_{6}$')
ax1.plot(T, sol[:,15], label=r'$e_{7}$')
ax1.plot(T, sol[:,16], label=r'$e_{8}$')
ax1.plot(T, sol[:,17], label=r'$e_{9}$')
ax1.plot(T, sol[:,18], label=r'$e_{10}$')
ax1.plot(T, sol[:,19], label=r'$e_{11}$')
ax1.plot(T, sol[:,20], label=r'$e_{12}$')
ax1.plot(T, sol[:,21], label=r'$e_{13}$')
ax1.plot(T, sol[:,22], label=r'$e_{14}$')
ax1.plot(T, sol[:,23], label=r'$e_{15}$')
ax1.plot(T, sol[:,24], label=r'$e_{16}$')
ax1.plot(T, sol[:,25], label=r'$e_{17}$')
ax1.plot(T, sol[:,26], label=r'$e_{18}$')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$e$')
ax1.legend()

ax1 = fig.add_subplot(514)
ax1.plot(T, sol[:,27], label=r'$j_{v_{1}}$')
ax1.plot(T, sol[:,28], label=r'$j_{v_{2}}$')
ax1.plot(T, sol[:,29], label=r'$j_{v_{3}}$')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$j_{v}$')
ax1.legend()

ax1 = fig.add_subplot(515)
ax1.plot(T, sol[:,30], label=r'$\lambda_{1}$')
ax1.plot(T, sol[:,31], label=r'$\lambda_{2}$')
ax1.plot(T, sol[:,32], label=r'$\lambda_{3}$')
ax1.plot(T, sol[:,33], label=r'$\lambda_{4}$')
ax1.plot(T, sol[:,34], label=r'$\lambda_{5}$')
ax1.plot(T, sol[:,35], label=r'$\lambda_{6}$')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$\lambda$')
ax1.legend()

plt.savefig('dgu_triangle_trajectory_new.png')