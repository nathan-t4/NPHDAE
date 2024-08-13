import jax.numpy as jnp
from scipy.integrate import odeint
from scipy.optimize import fsolve
import jax
from jax.experimental.ode import odeint

from helpers.integrator_factory import integrator_factory

import matplotlib.pyplot as plt

class DAESolver():
    """
    Solver for differential algebraic equations in semi-explicit form:
        dot{x} = f(x,y,t)
        0 = g(x,y,t)
    of index 1 (i.e. the Jacobian of g with respect to y has a nonzero determinant.)
    """
    def __init__(
            self, 
            f : callable, 
            g : callable, 
            num_diff_vars : int, 
            num_alg_vars : int):
        self.f = f
        self.g = g
        self.num_diff_vars = num_diff_vars
        self.num_alg_vars = num_alg_vars

        self.construct_coupled_odes()

    def construct_coupled_odes(self):
        """
        Construct the coupled system of ODEs that will be used to solve this DAE.
        """
        def gx(x,y,t,params):
            gg = lambda xx : self.g(xx, y, t, params)
            return jax.jacfwd(gg)(x)
        
        def gy(x,y,t,params):
            gg = lambda yy : self.g(x, yy, t, params)
            return jax.jacfwd(gg)(y)
        
        def gt(x,y,t,params):
            gg = lambda tt : self.g(x,y,tt,params)
            return jax.jacfwd(gg)(t)
        
        def construct_b(x,y,t,params):
            return jnp.matmul(gx(x,y,t,params), self.f(x,y,t,params)) + gt(x,y,t,params)
        
        # @jax.jit
        def y_dot(x,y,t,params):
            return - jnp.linalg.solve(gy(x,y,t,params), construct_b(x,y,t,params))
        
        # @jax.jit
        def f_coupled_system(z, t, params):
            x = z[0:self.num_diff_vars]
            y = z[self.num_diff_vars::]

            xp = self.f(x,y,t,params)
            yp = self.y_dot(x,y,t,params)

            return jnp.concatenate((xp, yp))

        self.y_dot = y_dot
        self.f_coupled_system = f_coupled_system

    def solve_dae(self, z0, T, params, y0_tol=1e-9):
        """
        Solve the DAE
        """

        x0 = z0[0:self.num_diff_vars]
        y0 = z0[self.num_diff_vars::]

        y0new, infodict, ier, mesg = fsolve(lambda yy : self.g(x0, yy, T[0], params), y0, full_output=True)
        
        if ier != 1:
            # throw an error if the algebraic states are not consistent.
            raise ValueError("Initial algebraic states were inconsistent. fsolve returned {}".format(mesg))

        if not (jnp.abs(y0new - y0) < y0_tol).all():
            print("Initial algebraic states {} were inconsistent. New initial algebraic state values are {}".format(y0, y0new))
            y0 = y0new
            z0 = jnp.concatenate((x0, y0))

        sol = odeint(self.f_coupled_system, z0, T, params)

        return sol
    
    def solve_dae_one_timestep_rk4(self, z0, t0, delta_t, params, y0_tol=1e-9):
        """
        Solve one forward timestep using RK4.
        Warning that this method does not check for consistency of the algebraic states.
        """      
        k1 = self.f_coupled_system(z0, t0, params)
        k2 = self.f_coupled_system(z0 + delta_t/2 * k1, t0 + delta_t/2, params)
        k3 = self.f_coupled_system(z0 + delta_t/2 * k2, t0 + delta_t/2, params)
        k4 = self.f_coupled_system(z0 + delta_t * k3, t0 + delta_t, params)

        return z0 + delta_t/6 * (k1 + 2 * k2 + 2 * k3 + k4)


def main():

    def f(x, y, t, params):
        return -x + y

    def g(x, y, t, params):
        return x + y - 1

    solver = DAESolver(f, g, num_diff_vars=1, num_alg_vars=1)

    diff_vars_init = jnp.array([0.0])
    alg_vars_init = jnp.array([1.0])

    z0 = jnp.concatenate((diff_vars_init, alg_vars_init))
    T = jnp.linspace(0, 1, 100)

    sol = solver.solve_dae(z0, T, params=None)

    zs = [z0]
    for t in T[1:]:
        next_z = solver.solve_dae_one_timestep_rk4(zs[-1], t, 0.01, params=None)
        zs.append(next_z)
        
    zs = jnp.array(zs)
    zs = jnp.sum(zs, axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sol[:,0] + sol[:,1], label='solve_dae')
    ax.plot(zs, label='rk4')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()