import numpy as np
import jax.numpy as jnp

def interpolate(t, t_vals, y_vals):
    """
    Linearly interpolate the solution at time t from the given solution values.

    Inputs:
    t: float, time at which to interpolate the solution
    t_vals: numpy array of shape (n,), time values
    y_vals: numpy array of shape (m, n), solution values at each time step

    Outputs:
    y: numpy array of shape (m,), interpolated solution at time t
    """
    assert t_vals.shape[0] == y_vals.shape[1]
    assert t_vals[0] <= t <= t_vals[-1]
    
    t_sub = t_vals - t
    i = np.argmax(t_sub >= 0) - 1

    if i == -1:
        i = 0

    y = y_vals[:, i] + (y_vals[:, i+1] - y_vals[:, i]) * ((t - t_vals[i]) / (t_vals[i+1] - t_vals[i]))

    return y