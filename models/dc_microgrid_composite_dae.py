import sys
sys.path.append('../')
from models.ph_dae import PHDAE
from models.composite_ph_dae import CompositePHDAE
import jax.numpy as jnp
import matplotlib.pyplot as plt

ph_dae_list = []

# DGU 1
AC = jnp.array([[0.0], [0.0], [1.0]])
AR = jnp.array([[1.0], [-1.0], [0.0]])
AL = jnp.array([[0.0], [1.0], [-1.0]])
AV = jnp.array([[1.0], [0.0], [0.0]])
AI = jnp.array([[0.0], [0.0], [-1.0]])

R = 0.2
L = 1.8e-3
C = 2.2e-3

def r_func(delta_V, params=None):
    return delta_V / R

def q_func(delta_V, params=None):
    return C * delta_V

def grad_H_func(phi, params=None):
    return phi / L

def u_func(t, params):
    return jnp.array([0.8, 100.0])

dgu1 = PHDAE(AC, AR, AL, AV, AI, grad_H_func, q_func, r_func, u_func)
ph_dae_list.append(dgu1)

# Transmission line
AC = jnp.array([[0.0], [0.0], [0.0]])
AR = jnp.array([[-1.0], [1.0], [0.0]])
AL = jnp.array([[0.0], [-1.0], [1.0]])
AV = jnp.array([[0.0], [0.0], [0.0]])
AI = jnp.array([[0.0], [0.0], [0.0]])

R = 0.05
L = 1.8e-6

def r_func(delta_V, params=None):
    return delta_V / R

def q_func(delta_V, params=None):
    return None

def grad_H_func(phi, params=None):
    return phi / L

def u_func(t, params):
    return jnp.array([])

transmission_line = PHDAE(AC, AR, AL, AV, AI, grad_H_func, q_func, r_func, u_func)
ph_dae_list.append(transmission_line)

# DGU 2
AC = jnp.array([[0.0], [0.0], [1.0]])
AR = jnp.array([[1.0], [-1.0], [0.0]])
AL = jnp.array([[0.0], [1.0], [-1.0]])
AV = jnp.array([[1.0], [0.0], [0.0]])
AI = jnp.array([[0.0], [0.0], [-1.0]])

R = 0.2
L = 1.8e-3
C = 2.2e-3

def r_func(delta_V, params=None):
    return delta_V / R

def q_func(delta_V, params=None):
    return C * delta_V

def grad_H_func(phi, params=None):
    return phi / L

def u_func(t, params):
    return jnp.array([1.1, 100.0])

dgu2 = PHDAE(AC, AR, AL, AV, AI, grad_H_func, q_func, r_func, u_func)
ph_dae_list.append(dgu2)

A_lambda = jnp.array([
    [0.0, 0.0], 
    [0.0, 0.0],
    [1.0, 0.0],
    [-1.0, 0.0],
    [0.0, 0.0],
    [0.0, -1.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 1.0],
])
composite_dae = CompositePHDAE(ph_dae_list, A_lambda)

x0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
y0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
z0 = jnp.concatenate((x0, y0))
T = jnp.linspace(0, 1.5, 1000)

sol = composite_dae.solve(z0, T, params_list=[None, None, None])

print(sol.shape)

fig = plt.figure()

ax1 = fig.add_subplot(631)
ax1.plot(T, sol[:,0])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('q')

ax1 = fig.add_subplot(632)
ax1.plot(T, sol[:,1])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('phi')

ax1 = fig.add_subplot(633)
ax1.plot(T, sol[:,2])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('e1')

ax1 = fig.add_subplot(634)
ax1.plot(T, sol[:,3])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('e2')

ax1 = fig.add_subplot(635)
ax1.plot(T, sol[:,4])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('e3')

ax1 = fig.add_subplot(636)
ax1.plot(T, sol[:,5])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

ax1 = fig.add_subplot(637)
ax1.plot(T, sol[:,6])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

ax1 = fig.add_subplot(638)
ax1.plot(T, sol[:,7])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

ax1 = fig.add_subplot(639)
ax1.plot(T, sol[:,8])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

ax1 = fig.add_subplot(6,3,10)
ax1.plot(T, sol[:,9])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

ax1 = fig.add_subplot(6,3,11)
ax1.plot(T, sol[:,10])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

ax1 = fig.add_subplot(6,3,12)
ax1.plot(T, sol[:,11])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

ax1 = fig.add_subplot(6,3,13)
ax1.plot(T, sol[:,12])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

ax1 = fig.add_subplot(6,3,14)
ax1.plot(T, sol[:,13])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

ax1 = fig.add_subplot(6,3,15)
ax1.plot(T, sol[:,14])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

ax1 = fig.add_subplot(6,3,16)
ax1.plot(T, sol[:,15])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

ax1 = fig.add_subplot(6,3,17)
ax1.plot(T, sol[:,16])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

ax1 = fig.add_subplot(6,3,18)
ax1.plot(T, sol[:,17])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

plt.savefig('dc_microgrid_trajectory.png')