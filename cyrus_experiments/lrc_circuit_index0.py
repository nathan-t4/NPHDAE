

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

    # def f_solve_alg_eq(y):
    #     return g(diff_vars_init, y, t0)
    
    # alg_vars_init = fsolve(f_solve_alg_eq, alg_vars_init)

    # def gx(x,y,t):
    #     gg = lambda xx : g(xx, y, t)
    #     return jax.jacfwd(gg)(x)
    
    # def gy(x,y,t):
    #     gg = lambda yy : g(x, yy, t)
    #     return jax.jacfwd(gg)(y)
    
    # # def gx_f_jvp(x,y,t):
    # #     gg = lambda xx : g(xx, y, t)
    # #     return jax.jvp(gg, x, f(x,y,t))
    
    # def gt(x,y,t):
    #     gg = lambda tt : g(x,y,tt)
    #     return jax.jacfwd(gg)(t)
    
    # def construct_b(x,y,t):
    #     return jnp.matmul(gx(x,y,t), f(x,y,t)) + gt(x,y,t)
    
    # def ydot(x,y,t):
    #     return - jnp.linalg.solve(gy(x,y,t), construct_b(x,y,t))
    
    # def f_coupled_system(z, t):
    #     x = z[0:2]
    #     y = z[2::]

    #     xp = f(x,y,t)
    #     yp = ydot(x,y,t)

    #     return jnp.concatenate((xp, yp))

    # sol = odeint(f_coupled_system, z0, T)

    solver = DAESolver(f, g, 2, 4)
    sol = solver.solve_dae(z0, T, params=None)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sol[:,5])
    plt.show()

if __name__ == "__main__":
    main()