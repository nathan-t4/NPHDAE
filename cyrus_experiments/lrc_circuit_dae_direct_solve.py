

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve

import matplotlib.pyplot as plt

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

def direct_solve(
        differential_eqs : callable,
        algebraic_eqs : callable,
        diff_vars_init : np.ndarray,
        alg_vars_init : np.ndarray,
        integrator : callable,
        params : np.ndarray,
        voltage_function : callable,
        t0 : float = 0.0,
        delta_t : float = 1e-3,
        num_t_steps : int = 100,
    ):
    """
    Solve the DAE using a dynamic iteration scheme.
    """

    diff_vars = [diff_vars_init]
    t_vals = [t0]

    def f_alg_eqns(y):
        return algebraic_eqs(diff_vars_init, y, t0, params, voltage_function)
    
    alg_vars_init = fsolve(f_alg_eqns, alg_vars_init)

    alg_vars = [alg_vars_init]

    for t_ind in range(num_t_steps):

        deriv_diff_vars = differential_eqs(diff_vars[-1], alg_vars[-1], t_vals[-1], params)
        next_diff_vars = diff_vars[-1] + delta_t * deriv_diff_vars

        diff_vars.append(next_diff_vars)
        t_vals.append(t_vals[-1] + delta_t)

        def f_alg_eqns(y):
            return algebraic_eqs(next_diff_vars, y, t_vals[-1], params, voltage_function)

        next_alg_vars = fsolve(f_alg_eqns, alg_vars[-1])

        alg_vars.append(next_alg_vars)

    return diff_vars, alg_vars
        

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

    diff_vars_out, alg_vars_out = direct_solve(
                                    differential_eqs,
                                    algebraic_eqs,
                                    diff_vars_init,
                                    alg_vars_init,
                                    None,
                                    params,
                                    voltage_source,
                                    t0=0.0,
                                    delta_t=0.01,
                                    num_t_steps=1000,
                                )
    
    print(diff_vars_out)
    print(alg_vars_out)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.array(alg_vars_out)[:, 3])

    plt.show()

if __name__ == "__main__":
    main()