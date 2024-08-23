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

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(gnorm)

plt.savefig('dgu_triangle_gnorm.png')

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
ax1.set_ylabel(r'$q_{3}$')

ax1 = fig.add_subplot(634)
ax1.plot(T, sol[:,3])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$\phi_{1}$')

ax1 = fig.add_subplot(635)
ax1.plot(T, sol[:,4])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$\phi_{2}$')

ax1 = fig.add_subplot(636)
ax1.plot(T, sol[:,5])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$\phi_{3}$')

ax1 = fig.add_subplot(637)
ax1.plot(T, sol[:,6])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$\phi_{4}$')

ax1 = fig.add_subplot(638)
ax1.plot(T, sol[:,7])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$\phi_{5}$')

ax1 = fig.add_subplot(639)
ax1.plot(T, sol[:,8])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$\phi_{6}$')

ax1 = fig.add_subplot(6,3,10)
ax1.plot(T, sol[:,9])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$e_{1}$')

ax1 = fig.add_subplot(6,3,11)
ax1.plot(T, sol[:,10])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$e_{2}$')

ax1 = fig.add_subplot(6,3,12)
ax1.plot(T, sol[:,11])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$e_{3}$')

ax1 = fig.add_subplot(6,3,13)
ax1.plot(T, sol[:,12])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$e_{4}$')

ax1 = fig.add_subplot(6,3,14)
ax1.plot(T, sol[:,13])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$e_{5}$')

ax1 = fig.add_subplot(6,3,15)
ax1.plot(T, sol[:,13])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$e_{6}$')

ax1 = fig.add_subplot(6,3,16)
ax1.plot(T, sol[:,13])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$e_{7}$')

# ax1 = fig.add_subplot(6,3,15)
# ax1.plot(T, sol[:,14])
# ax1.set_xlabel('Time [s]')
# ax1.set_ylabel(r'$j_{v_{1}}$')

# ax1 = fig.add_subplot(6,3,16)
# ax1.plot(T, sol[:,15])
# ax1.set_xlabel('Time [s]')
# ax1.set_ylabel(r'$j_{v_{2}}$')

ax1 = fig.add_subplot(6,3,17)
ax1.plot(T, sol[:,-2])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$\lambda_{5}$')

ax1 = fig.add_subplot(6,3,18)
ax1.plot(T, sol[:,-1])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$\lambda_{6}$')


plt.savefig('dgu_triangle_trajectory.png')