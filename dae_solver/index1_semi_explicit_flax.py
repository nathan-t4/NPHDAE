import jax.numpy as jnp
from scipy.integrate import odeint
from scipy.optimize import fsolve
import jax
from jax.experimental.ode import odeint
import flax.linen as nn
import optax

import jaxopt

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
            diff_indices : list, 
            alg_indices : list):
        self.f = f
        self.g = g
        self.diff_indices = diff_indices
        self.alg_indices = alg_indices

        self.construct_coupled_odes()
        self.construct_optimizer()

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
        
        def y_dot(x,y,t,params):
            return - jnp.linalg.solve(gy(x,y,t,params), construct_b(x,y,t,params))
        
        def f_coupled_system(z, t, params):
            x = z[0 : len(self.diff_indices)]
            y = z[len(self.diff_indices) ::]

            xp = self.f(x,y,t,params)
            yp = self.y_dot(x,y,t,params)

            return jnp.concatenate((xp, yp))

        self.y_dot = y_dot
        self.f_coupled_system = f_coupled_system

    def construct_optimizer(self):
        self.optimizer = optax.adam(1e-2)

    def optimize(self, f, init_params):
        opt_state = self.optimizer.init(init_params)

        def step(params, opt_state):
            f_value, grads = jax.value_and_grad(f)(params)
            update, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, update)
            return params, f_value, opt_state
        
        params = init_params
        for k in range(10):
            params, f_value, opt_state = step(params, opt_state)

        return params, f_value

    def solve_dae(self, z0, T, params, y0_tol=1e-9):
        """
        Solve the DAE
        """

        x0 = z0[:len(self.diff_indices)]
        y0 = z0[len(self.diff_indices) ::]

        # y0new, infodict, ier, mesg = fsolve(lambda yy : self.g(x0, yy, T[0], params), y0, full_output=True)
        
        # if ier != 1:
        #     # throw an error if the algebraic states are not consistent.
        #     raise ValueError("Initial algebraic states were inconsistent. fsolve returned {}".format(mesg))
        
        # fsolve = jaxopt.ScipyRootFinding(
        #     dtype=jnp.float32,
        #     method='lm', 
        #     optimality_fun=lambda yy : self.g(x0, yy, T[0], params),
        #     tol=y0_tol)
        
        # solver = fsolve.run(init_params=y0)
        # y0new = solver.params

        y0new, g = self.optimize(lambda yy : optax.squared_error(self.g(x0, yy, T[0], params)).mean(), init_params=y0) # MSE

        # if not (jnp.abs(y0new - y0) < y0_tol).all():
        # print("Initial algebraic states {} were inconsistent. New initial algebraic state values are {}".format(y0, y0new))
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

    solver = DAESolver(f, g, diff_indices=1, alg_indices=1)

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