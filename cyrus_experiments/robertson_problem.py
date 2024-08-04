import jax.numpy as jnp

import numpy as np
import matplotlib.pyplot as plt
from scipy_dae.integrate import solve_dae


def F(t, y, yp):
    """Define implicit system of differential algebraic equations."""
    y1, y2, y3 = y
    y1p, y2p, y3p = yp

    F = np.zeros(3, dtype=y.dtype)
    F[0] = y1p - (-0.04 * y1 + 1e4 * y2 * y3)
    F[1] = y2p - (0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2)
    F[2] = y1 + y2 + y3 - 1 # algebraic equation

    return F


# time span
t0 = 0
t1 = 1e7
t_span = (t0, t1)
t_eval = np.logspace(-6, 7, num=1000)

# initial conditions
y0 = np.array([1, 0, 0], dtype=float)
yp0 = np.array([-0.04, 0.04, 0], dtype=float)

# solver options
# method = "Radau"
method = "BDF" # alternative solver
atol = rtol = 1e-6

# solve DAE system
sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, t_eval=t_eval)
t = sol.t
y = sol.y

# visualization
fig, ax = plt.subplots()
ax.set_xlabel("t")
ax.plot(t, y[0], label="y1")
ax.plot(t, y[1] * 1e4, label="y2 * 1e4")
ax.plot(t, y[2], label="y3")
ax.set_xscale("log")
ax.legend()
ax.grid()
plt.show()