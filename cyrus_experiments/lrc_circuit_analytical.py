# code to simulate an LRC circuit using the analytical solution with an external voltage source

import numpy as np
import matplotlib.pyplot as plt

# import ode library
from scipy.integrate import odeint


def dynamics_func(y, t, R, L, C, voltage_source):
    q, i = y
    dq_dt = i
    di_dt = (voltage_source(t) - R * i - q / C) / L
    return [dq_dt, di_dt]

if __name__ == "__main__":
    # initial conditions
    y0 = [0, 0]

    # define the circuit parameters
    R = 1
    L = 1
    C = 1

    # define the second order differential equation with respect to time
    def voltage_source(t):
        return np.sin(30 * t)

    # time span
    t = np.linspace(0, 1.5, 10000)

    def f(y, t):
        return dynamics_func(y, t, R, L, C, voltage_source)

    # solve the differential equation
    sol = odeint(f, y0, t)

    # plot the results
    plt.plot(t, sol[:, 1])
    plt.xlabel("Time")
    plt.ylabel("")
    plt.legend()
    plt.show()