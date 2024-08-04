

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve

def differential_eqs(diff_vars, alg_vars, t, params):
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
    q, phi = diff_vars
    e1, e2, e3, jv = alg_vars
    R, L, C = params

    d_q = phi / L
    d_phi = e2 - e3

    return np.array([d_q, d_phi])

def algebraic_eqs(diff_vars, alg_vars, t, params, voltage_source):
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
    q, phi = diff_vars
    e1, e2, e3, jv = alg_vars
    R, L, C = params

    eq1 = jv - (e1 - e2) / R
    eq2 = (e1 - e2) / R - phi / L
    eq3 = q - C * e3
    eq4 = e1 - voltage_source(t)

    return np.array([eq1, eq2, eq3, eq4])

def dynamic_iteration_scheme_one_timestep(
        differential_eqs : callable,
        algebraic_eqs : callable,
        diff_vars_init : np.ndarray,
        alg_vars_init : np.ndarray,
        integrator : callable,
        t : float,
        params : np.ndarray,
        voltage_function : callable,
        delta_t : float = 1e-3,
        l_max : int = 100
    ):
    """
    Solve the DAE using a dynamic iteration scheme.
    """

    derivative_diff_vars_curr = differential_eqs(diff_vars_init, alg_vars_init, t, params)

    diff_vars_iters = [diff_vars_init]
    alg_vars_iters = [alg_vars_init]
    derivative_diff_vars_iters = [derivative_diff_vars_curr]

    for iter in range(l_max):

        diff_vars_next = diff_vars_init + delta_t * derivative_diff_vars_curr
        derivative_diff_vars_next = differential_eqs(diff_vars_next, alg_vars_iters[-1], t + delta_t, params)

        def alg_vars_fsolve(alg_vars):
            return algebraic_eqs(diff_vars_next, alg_vars, t + delta_t, params, voltage_function)

        alg_vars_next = fsolve(alg_vars_fsolve, alg_vars_iters[-1])

        diff_vars_iters.append(diff_vars_next)
        alg_vars_iters.append(alg_vars_next)
        derivative_diff_vars_iters.append(derivative_diff_vars_next)

    return diff_vars_iters[-1], alg_vars_iters[-1], derivative_diff_vars_iters[-1]

def main():
    # initial conditions
    diff_vars_init = np.array([0, 0])
    alg_vars_init = np.array([0, 0, 0, 0])

    # define the circuit parameters
    R = 1
    L = 1
    C = 1

    params = np.array([R, L, C])

    # define the second order differential equation with respect to time
    def voltage_source(t):
        return np.sin(30 * t)
    
    # time span
    t = np.linspace(0, 1.5, 1000)

    print(dynamic_iteration_scheme_one_timestep(
        differential_eqs,
        algebraic_eqs,
        diff_vars_init,
        alg_vars_init,
        None,
        0,
        params,
        voltage_source,
        0.001,
        l_max=1000
    ))

if __name__ == "__main__":
    main()