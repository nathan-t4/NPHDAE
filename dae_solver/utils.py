import jax
import jax.random as random
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

@jax.jit
def randomized_svd(A, k, random_state=0, n_oversamples=10):
    """
    Perform randomized truncated SVD on matrix A.

    Args:
        A: Input matrix of shape (m, n).
        k: Number of desired singular values and vectors.
        random_state: Seed for random number generation (default: 0).
        n_oversamples: Number of random projections for oversampling (default: 10).

    Returns:
        U: Left singular vectors of shape (m, k).
        S: Singular values of shape (k,).
        Vh_B: Right singular vectors of shape (k, n_oversamples).
    """
    m, n = A.shape
    rng = random.PRNGKey(random_state)
    random_matrix = random.normal(rng, (n, n_oversamples))

    Y = jnp.dot(A, random_matrix)
    Q, _ = jnp.linalg.qr(Y)
    B = jnp.dot(Q.T, A)
    U, S, Vh_B = jnp.linalg.svd(B, full_matrices=False)
    U = jnp.dot(Q, U)

    print(U.shape, S.shape, Vh_B.shape)

    return U, S, Vh_B

@jax.jit
def truncated_svd(A, k):
    """
    Wrapper for randomized_svd to perform truncated SVD.

    Args:
        A: Input matrix of shape (m, n).
        k: Number of singular values and vectors to compute.

    Returns:
        U: Left singular vectors of shape (m, k).
        S: Singular values of shape (k,).
        Vh: Right singular vectors of shape (k, n).
    """
    return randomized_svd(A, 3)