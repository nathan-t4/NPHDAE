import jax.numpy as jnp
from scipy.integrate import odeint
from scipy.optimize import fsolve
import jax
from jax.experimental.ode import odeint

import matplotlib.pyplot as plt

class RandomDAESolver():
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
        def gx(x,y,t,jax_key,params):
            gg = lambda xx : self.g(xx, y, t, jax_key, params)
            return jax.jacfwd(gg)(x)
        
        def gy(x,y,t,jax_key,params):
            gg = lambda yy : self.g(x, yy, t, jax_key, params)
            return jax.jacfwd(gg)(y)
        
        def gt(x,y,t,jax_key,params):
            gg = lambda tt : self.g(x,y,tt,jax_key,params)
            return jax.jacfwd(gg)(t)
        
        def construct_b(x,y,t,jax_key,params):
            return jnp.matmul(gx(x,y,t,jax_key,params), self.f(x,y,t,jax_key,params)) + gt(x,y,t,jax_key,params)
        
        @jax.jit
        def y_dot(x,y,t,jax_key,params):
            return - jnp.linalg.solve(gy(x,y,t,jax_key,params), construct_b(x,y,t,jax_key,params))
        
        @jax.jit
        def f_coupled_system(z, t, jax_key, params):
            x = z[0:self.num_diff_vars]
            y = z[self.num_diff_vars::]

            xp = self.f(x,y,t,jax_key,params)
            yp = self.y_dot(x,y,t,jax_key,params)

            return jnp.concatenate((xp, yp))

        self.y_dot = y_dot
        self.f_coupled_system = f_coupled_system

    def solve_dae(self, z0, T, jax_key, params, y0_tol=1e-9):
        """
        Solve the DAE
        """
        x0 = z0[0:self.num_diff_vars]
        y0 = z0[self.num_diff_vars::]

        y0new, infodict, ier, mesg = fsolve(lambda yy : self.g(x0, yy, T[0], jax_key, params), y0, full_output=True)
        
        if ier != 1:
            # throw an error if the algebraic states are not consistent.
            raise ValueError("Initial algebraic states were inconsistent. fsolve returned {}".format(mesg))

        if not (jnp.abs(y0new - y0) < y0_tol).all():
            print("Initial algebraic states {} were inconsistent. New initial algebraic state values are {}".format(y0, y0new))
            y0 = y0new
            z0 = jnp.concatenate((x0, y0))

        sol = odeint(self.f_coupled_system, z0, T, jax_key, params)

        return sol
    
    def solve_dae_one_timestep_rk4(self, z0, t0, delta_t, jax_key, params):
        """
        Solve one forward timestep using RK4.
        Warning that this method does not check for consistency of the algebraic states.
        """
        k1 = self.f_coupled_system(z0, t0, jax_key, params)
        k2 = self.f_coupled_system(z0 + delta_t/2 * k1, t0 + delta_t/2, jax_key, params)
        k3 = self.f_coupled_system(z0 + delta_t/2 * k2, t0 + delta_t/2, jax_key, params)
        k4 = self.f_coupled_system(z0 + delta_t * k3, t0 + delta_t, jax_key, params)

        return z0 + delta_t/6 * (k1 + 2 * k2 + 2 * k3 + k4)


def main():

    def f(x, y, t, jax_key, params):
        return -x + y

    def g(x, y, t, jax_key, params):
        return x + y - 1

    solver = RandomDAESolver(f, g, num_diff_vars=1, num_alg_vars=1)

    diff_vars_init = jnp.array([0.0])
    alg_vars_init = jnp.array([0.0])

    z0 = jnp.concatenate((diff_vars_init, alg_vars_init))
    T = jnp.linspace(0, 1, 100)

    key = jax.random.key(0)
    sol = solver.solve_dae(z0, T, key, params=None)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sol[:,0] + sol[:,1])
    plt.show()

if __name__ == "__main__":
    main()