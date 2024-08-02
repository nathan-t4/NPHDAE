import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import time
from functools import partial
from scipy_dae.integrate import solve_dae, consistent_initial_conditions
from scipy.optimize._numdiff import approx_derivative

def F(t, y, yp, u, A, system_data, splits):
    """Define implicit system of differential algebraic equations."""
    R, L, C = system_data
    AC, AR, AL, AV, AI = A

    y = np.nan_to_num(y)
    yp = np.nan_to_num(yp)
    # assert not np.isnan(y).any()
    # assert not np.isnan(yp).any()
    # assert not np.isinf(y).any()
    # assert not np.isinf(yp).any()
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

    F0 = AC @ qp + AL @ dH1 + AR @ g(e) + AI @ i + (AV @ jv if len(jv) > 0 else 0) 
    F1 = phip - AL.T @ e
    F2 = AC.T @ e - dH0
    F3 = AV.T @ e - v         
    equations = np.concatenate((F0, F1, F2, F3)) if len(jv) > 0 else np.concatenate((F0, F1, F2))
    # print(equations)
    return equations

def jac(t, y, yp, u, A, system_data, splits, f=None):
    n = len(y)
    z = np.concatenate((y, yp))

    def fun_composite(t, z):
        y, yp = z[:n], z[n:]
        func = partial(F, u=u, A=A, system_data=system_data, splits=splits)
        return func(t, y, yp)
    
    J = approx_derivative(lambda z: fun_composite(t, z), 
                            z, method="2-point", f0=f)
    J = J.reshape((n, 2 * n))
    Jy, Jyp = J[:, :n], J[:, n:]
    return Jy, Jyp

# def f(t, z):
#     n = len(y)
#     y, yp = z[:n], z[n:]
#     return np.concatenate((yp, F(t, y, yp)))

def get_microgrid_params():
    AC = np.array([[-1, 0, 0, 1]]).T
    AR = np.array([[0, -1, 1, 0]]).T
    AL = np.array([[0, 0, -1, 1]]).T
    AV = np.array([[-1, 1, 0, 0]]).T
    AI = np.array([[0, 0, 0, 0]]).T
    R = np.array(1.0)
    L = np.array(1.0)
    C = np.array(1.0)
    A = (AC, AR, AL, AV, AI)
    system_data = (R, L, C)
    y0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    yp0 = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    init_conditions = (y0, yp0)
    return (A, system_data, init_conditions)

def get_microgrid_policy(t):
    return np.array([[0, np.sin(t)]]).T # [i, v]

def get_lc1_params():
    AC = np.array([[-1, 1]]).T
    AR = np.array([[0, 0]]).T
    AL = np.array([[1, -1]]).T
    AV = np.array([[0, 0]]).T
    AI = np.array([[0, 0]]).T
    R = np.array(1.0)
    L = np.array(1.0)
    C = np.array(1.0)
    A = (AC, AR, AL, AV, AI)
    system_data = (R, L, C)
    y0 = np.array([1.0, 0.0, 0.0, 1.0])
    yp0 = np.array([0.0, -1.0, 0.0, 0.0])
    init_conditions = (y0, yp0)
    return (A, system_data, init_conditions)

def get_lc1_policy(t):
    return np.array([[0, 0]]).T

# time span
t0 = 0
t1 = 1e2
t_span = (t0, t1)
t_eval = np.linspace(0, 1e2, num=1000)

# solver options
method = "Radau"
# method = "BDF" # alternative solver
atol = rtol = 1e-4

# system parameters
system = get_microgrid_params()
u = get_microgrid_policy
# system = get_lc1_params()
# u = get_lc1_policy
A, system_data, init_conditions = system
AC, AR, AL, AV, AI = A
splits = np.array([len(AC.T), 
                   len(AC.T) + len(AL.T),
                   len(AC.T) + len(AL.T) + len(AC)])
y0, yp0 = init_conditions
func = partial(F, A=A, system_data=system_data, u=u, splits=splits)
jac = partial(jac, u=u, A=A, system_data=system_data, splits=splits)
f0 = func(0.0, y0, yp0)
print(f"f0: {f0}")
Jy0, Jyp0 = jac(t0, y0, yp0)
J0 = Jy0 + Jyp0
print(f"Jy0:\n{Jy0}")
print(f"Jyp0:\n{Jyp0}")
print(f"J0:\n{Jyp0}")
print(f"rank(Jy0):  {np.linalg.matrix_rank(Jy0)}")
print(f"rank(Jyp0): {np.linalg.matrix_rank(Jyp0)}")
print(f"rank(J0):   {np.linalg.matrix_rank(J0)}")
print(f"J0.shape: {J0.shape}")
# print('IC test', )
# y0, yp0, fnorm = consistent_initial_conditions(func, jac, t0, y0, yp0)

# solve DAE
start = time.time()
sol = solve_dae(func, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, t_eval=t_eval)
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
# ax.plot(t, y[6,:], label="jv")
ax.legend()
ax.grid()
plt.show()