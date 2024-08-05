

import jax.numpy as jnp
from scipy.integrate import odeint
from scipy.optimize import fsolve
import jax
from jax.experimental.ode import odeint

import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from dae_solver.index1_semi_explicit import DAESolver


# Define parameters
R = 1
C = 1
L = 1

def f(x, y, t, params):
    """
    Define the differential equations for the LRC circuit.
    
    Parameters
    ----------
    diff_vars : np.ndarray
        The differential variables.
    alg_vars : np.ndarray
        The algebraic variables.
    t : float
        The time.
    
    Returns
    -------
    np.ndarray
        The time derivatives of the differential equation.
    """
    q, phi = x
    e1, e2, e3, jv = y

    d_q = phi / L
    d_phi = e2 - e3

    return jnp.array([d_q, d_phi])

def g(x, y, t, params):
    """
    Define the algebraic equations for the LRC circuit.

    Parameters
    ----------
    diff_vars : np.ndarray
        The differential variables.
    alg_vars : np.ndarray
        The algebraic variables.
    t : float
        The time.
    
    Returns
    -------
    np.ndarray
        The residuals of the algebraic equations.
    """
    q, phi = x
    e1, e2, e3, jv = y

    eq1 = jv - (e1 - e2) / R
    eq2 = (e1 - e2) / R - phi / L
    eq3 = q - C * e3
    eq4 = e1 - jnp.sin(30 * t)

    return jnp.array([eq1, eq2, eq3, eq4])
        

def main():
    # initial conditions
    diff_vars_init = jnp.array([0.0, 0.0])
    alg_vars_init = jnp.array([0.0, 0.0, 0.0, 0.0])

    z0 = jnp.concatenate((diff_vars_init, alg_vars_init))
    T = jnp.linspace(0, 1.5, 1000)
    dt = T[2] - T[1]

    solver = DAESolver(f, g, 2, 4)
    # sol = solver.solve_dae(z0, T, params=None)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(sol[:,5])
    # plt.show()

    sol = solver.solve_dae(z0, T, None)

    g_vals = []
    for t_ind in range(sol.shape[0]):
        t = T[t_ind]
        z = sol[t_ind, :]
        x = z[0:2]
        y = z[2::]
        g_vals.append(solver.g(x,y,t,None))

    g_vals = jnp.array(g_vals)

    print(g_vals)

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(sol[:,2])
    ax = fig.add_subplot(212)
    ax.plot([sol[t_ind, 2] - jnp.sin(30 * t_ind * dt) for t_ind in range(len(T))])
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(411)
    ax.plot(g_vals[:, 0])

    ax = fig.add_subplot(412)
    ax.plot(g_vals[:, 1])

    ax = fig.add_subplot(413)
    ax.plot(g_vals[:, 2])

    ax = fig.add_subplot(414)
    ax.plot(g_vals[:, 3])

    plt.show()

if __name__ == "__main__":
    main()