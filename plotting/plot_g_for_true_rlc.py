import jax.numpy as jnp
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from models.ph_dae import PHDAE

AC = jnp.array([[0.0], [0.0], [1.0]])
AR = jnp.array([[1.0], [-1.0], [0.0]])
AL = jnp.array([[0.0], [1.0], [-1.0]])
AV = jnp.array([[1.0], [0.0], [0.0]])
AI = jnp.array([[0.0], [0.0], [0.0]])

R = 1
L = 1
C = 1

x0 = jnp.array([0.0, 0.0])
y0 = jnp.array([0.0, 0.0, 0.0, 0.0])
z0 = jnp.concatenate((x0, y0))
T = jnp.linspace(0, 1.5, 1000)
dt = T[2] - T[1]

def r_func(delta_V, params=None):
    return delta_V / R

def q_func(delta_V, params=None):
    return C * delta_V

def grad_H_func(phi, params=None):
    return phi / L

def u_func(t, params):
    return jnp.array([jnp.sin(30 * t)])

dae = PHDAE(AC, AR, AL, AV, AI, grad_H_func, q_func, r_func, u_func)

sol = dae.solve(z0, T, None)

g_vals = []
g_vals_squared = []
for t_ind in range(sol.shape[0]):
    t = T[t_ind]
    z = sol[t_ind, :]
    x = z[0:2]
    y = z[2::]
    g_vals.append(dae.solver.g(x,y,t,None))
    g_vals_squared.append(jnp.sum(dae.solver.g(x,y,t,None)**2))

g_vals = jnp.array(g_vals)
g_vals_squared = jnp.array(g_vals_squared)

print(g_vals)

# fig = plt.figure()
# ax = fig.add_subplot(211)
# ax.plot(sol[:,2])
# ax = fig.add_subplot(212)
# ax.plot([sol[t_ind, 2] - u_func(t_ind * dt, None) for t_ind in range(len(T))])

# fig = plt.figure()
# ax = fig.add_subplot(411)
# ax.plot(g_vals[:, 0])

# ax = fig.add_subplot(412)
# ax.plot(g_vals[:, 1])

# ax = fig.add_subplot(413)
# ax.plot(g_vals[:, 2])

# ax = fig.add_subplot(414)
# ax.plot(g_vals[:, 3])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(g_vals_squared)

plt.savefig('g_for_true_rlc.png')