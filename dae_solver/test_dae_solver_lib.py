import jax.numpy as jnp

import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from scipy_dae.integrate import solve_dae


@jax.jit
def F(t, y, yp, params=jnp.array([0.1, 0.01])):
    """Define implicit system of differential algebraic equations."""

    # Example RC circuit from this page: https://en.wikipedia.org/wiki/Modified_nodal_analysis#:~:text=In%20electrical%20engineering%2C%20modified%20nodal,but%20also%20some%20branch%20currents.

    C, G = params

    A = jnp.array([
        [G, -G, 1,],
        [-G, G, 0],
        [1, 0, 0]
    ])

    E = jnp.array([
        [0, 0, 0],
        [0, C, 0],
        [0, 0, 0]
    ])

    f = jnp.array([0, 0, jnp.sin(3 * t)])

    F = E @ yp + A @ y - f
    return F


# time span
t0 = 0
t1 = 1.5
t_span = (t0, t1)
t_eval = jnp.linspace(t0, t1, num=1000)

# initial conditions
y0 = jnp.array([0, 0, 0], dtype=float)
yp0 = jnp.array([0, 0, 0], dtype=float)

# solver options
method = "Radau"
# method = "BDF" # alternative solver
atol = rtol = 1e-6

# solve DAE system
sol = solve_dae(F, t_span, y0, yp0, atol=atol, dense_output=True, rtol=rtol, method=method, t_eval=t_eval)
t = sol.t
y = sol.y

# visualization
fig = plt.figure()

ax = fig.add_subplot(311)
ax.plot(t, y[0], label="voltage at source")

ax = fig.add_subplot(312)
ax.plot(t, y[1], label="voltage at capacitor")

ax = fig.add_subplot(313)
ax.plot(t, y[2], label="current through voltage source")

# ax.legend()
# ax.grid()
plt.show()