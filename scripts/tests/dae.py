import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import time
from functools import partial
from scipy_dae.integrate import solve_dae, consistent_initial_conditions
from scipy.optimize._numdiff import approx_derivative

def F(t, y, yp, u, A, system_data):
    """Define implicit system of differential algebraic equations."""
    R, L, C, splits = system_data
    AC, AR, AL, AV, AI = A

    y = np.nan_to_num(y)
    yp = np.nan_to_num(yp)
    qc, phi, e, jv = np.split(y, splits)
    qp, phip, ep, jvp = np.split(yp, splits)
    i, v = u(t)
    g = lambda e : (AR.T @ e) / R 

    def H(y):
        qc, phi, e, jv = np.split(y, splits)
        return 0.5 * ((L**(-1)) * phi.T @ phi + (C**(-1)) * qc.T @ qc)
     
    # dH = jax.grad(H)(y)
    # dH = np.split(dH, splits)
    dH1 = phi / L
    dH0 = qc / C

    F0 = AC @ qp + AL @ dH1 + AV @ jv + AR @ g(e) + AI @ i 
    F1 = phip - AL.T @ e
    F2 = AC.T @ e - dH0                   # algebraic equation
    F3 = AV.T @ e - v                     # algebraic equation
    return np.concatenate((F0, F1, F2, F3))

def jac(t, y, yp, u, A, system_data, f=None):
    n = len(y)
    z = np.concatenate((y, yp))

    def fun_composite(t, z):
        y, yp = z[:n], z[n:]
        return F(t, y, yp, u, A, system_data)
    
    J = approx_derivative(lambda z: fun_composite(t, z), 
                            z, method="2-point", f0=f)
    J = J.reshape((n, 2 * n))
    Jy, Jyp = J[:, :n], J[:, n:]
    return Jy, Jyp

def f(t, z):
    n = len(y)
    y, yp = z[:n], z[n:]
    return np.concatenate((yp, F(t, y, yp)))

def get_microgrid_params():
    AC = np.array([[1, 0, 0, -1]]).T
    AR = np.array([[0, -1, 1, 0]]).T
    AL = np.array([[0, 0, -1, 1]]).T
    AV = np.array([[-1, 1, 0, 0]]).T
    AI = np.array([[0, 0, 0, 0]]).T
    splits = np.array([len(AC.T), 
                       len(AC.T) + len(AL.T), 
                       len(AC.T) + len(AL.T) + len(AC)])
    R = np.array(1.0)
    L = np.array(1.0)
    C = np.array(1.0)
    A = (AC, AR, AL, AV, AI)
    system_data = (R, L, C, splits)
    return (A, system_data)

def get_microgrid_policy(t):
    return np.array([[0, np.sin(t)]]).T # [i, v]

def get_lc1_params():
    AC = np.array([[-1, -1], [1, 0], [0, 1]])
    AR = np.array([[0, 0, 0]]).T
    AL = np.array([[0, -1, 1]])

# time span
t0 = 0
t1 = 1e3
t_span = (t0, t1)
t_eval = np.linspace(0, 1e2, num=1000)

# solver options
# method = "Radau"
method = "Radau" # alternative solver
atol = rtol = 1e-4

# system parameters
A, system_data = get_microgrid_params()
AC, AR, AL, AV, AI = A
len_y = len(AC.T) + len(AL.T) + len(AC) + len(AV.T)
# initial conditions
u = get_microgrid_policy
func = partial(F, A=A, system_data=system_data, u=u)
jac = partial(jac, A=A, system_data=system_data, u=u)
y0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# yp0 = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
yp0 = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
print(func(0.0, y0, yp0))
# y0, yp0, fnorm = consistent_initial_conditions(func, jac, t0, y0, yp0)
# solve DAE
start = time.time()
sol = solve_dae(func, t_span, y0, yp0, atol=atol, rtol=rtol, method=method) # , t_eval=t_eval)
end = time.time()
print(f'elapsed time: {end - start}')
t = sol.t
y = sol.y
print('y shape', y.shape)
# visualization
fig, ax = plt.subplots()
ax.set_xlabel("t")
ax.plot(t, y[0,:], label="q")
ax.plot(t, y[1,:], label="phi")
# ax.plot(t, y[2,:], label="e1")
# ax.plot(t, y[3,:], label="e2")
# ax.plot(t, y[4,:], label="e3")
# ax.plot(t, y[5,:], label="e4")
ax.plot(t, y[6,:], label="jv")
ax.legend()
ax.grid()
plt.show()