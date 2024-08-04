

import jax.numpy as jnp
from scipy.integrate import odeint
from scipy.optimize import fsolve
import jax
from jax.experimental.ode import odeint

import matplotlib.pyplot as plt


def f(x, y, t):
    return -x + y

def g(x, y, t):
    return x + y - 1
        

def main():
    # initial conditions
    diff_vars_init = jnp.array([0.0])
    alg_vars_init = jnp.array([0.0])
    t0 = 0.0

    def f_solve_alg_eq(y):
        return g(diff_vars_init, y, t0)
    
    alg_vars_init = fsolve(f_solve_alg_eq, alg_vars_init)

    def gx(x,y,t):
        gg = lambda xx : g(xx, y, t)
        return jax.jacfwd(gg)(x)
    
    def gy(x,y,t):
        gg = lambda yy : g(x, yy, t)
        return jax.jacfwd(gg)(y)
    
    # def gx_f_jvp(x,y,t):
    #     gg = lambda xx : g(xx, y, t)
    #     return jax.jvp(gg, x, f(x,y,t))
    
    def gt(x,y,t):
        gg = lambda tt : g(x,y,tt)
        return jax.jacfwd(gg)(t)
    
    def construct_b(x,y,t):
        return jnp.matmul(gx(x,y,t), f(x,y,t)) + gt(x,y,t)
    
    def ydot(x,y,t):
        return - jnp.linalg.solve(gy(x,y,t), construct_b(x,y,t))
    
    def f_coupled_system(z, t):
        x = z[0:1]
        y = z[1::]

        xp = f(x,y,t)
        yp = ydot(x,y,t)

        return jnp.concatenate((xp, yp))
    
    z0 = jnp.concatenate((diff_vars_init, alg_vars_init))
    T = jnp.linspace(0, 1.5, 10000)

    sol = odeint(f_coupled_system, z0, T)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sol[:,0] + sol[:,1])
    plt.show()

if __name__ == "__main__":
    main()