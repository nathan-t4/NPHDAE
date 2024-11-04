import jax.numpy as jnp
from scipy.integrate import odeint
from scipy.optimize import fsolve
import jax
from jax.experimental.ode import odeint
import jax.scipy.linalg as la
import matplotlib.pyplot as plt
from dae_solver.utils import truncated_svd
from dae_solver.implicit_solvers import *

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
            num_alg_vars : int,
            regularization_method : str,
            reg_param : float,
            one_timestep_solver : str):
        self.f = f
        self.g = g
        self.num_diff_vars = num_diff_vars
        self.num_alg_vars = num_alg_vars
        self.regularization_method = regularization_method
        self.reg_param = int(reg_param) if regularization_method == 'truncated_svd' else float(reg_param)

        self.one_timestep_solver = self.get_one_timestep_solver(one_timestep_solver)
    
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
            # ggy = jax.jacfwd(gg)(y)
            # jax.debug.print('svd {}', jnp.linalg.svd(ggy, full_matrices=False)[1])
            # [1.1797365  0.9043466  0.24289489 0.00251038]
            return jax.jacfwd(gg)(y)
        
        def gt(x,y,t,params):
            gg = lambda tt : self.g(x,y,tt,params)
            return jax.jacfwd(gg)(t)
        
        def construct_b(x,y,t,params):
            return jnp.matmul(gx(x,y,t,params), self.f(x,y,t,params)) + gt(x,y,t,params)
        
        @jax.jit
        def y_dot(x,y,t,params):
            gy_matrix = gy(x,y,t,params)
            if self.regularization_method == 'tikhanov':
                "Tikhanov regularization with lambda = self.reg_param"
                gy_rank = jnp.linalg.matmul(gy_matrix.T, gy_matrix) + self.reg_param*jnp.eye(len(y))
                return - jnp.linalg.solve(
                    gy_rank, 
                    jnp.linalg.matmul(gy_matrix.T, construct_b(x,y,t,params))
                )
            elif self.regularization_method == 'truncated_svd':
                "Truncated SVD algorithm removing the smallest singular value"
                # raise NotImplementedError
                U, S, Vh_B = jnp.linalg.svd(gy_matrix,self.reg_param)
                s_vals = 1 / S
                s_vals = s_vals.at[-1].set(0.0)
                return -Vh_B.T @ jnp.diag(s_vals) @ U.T @ construct_b(x,y,t,params)
            
            return - jnp.linalg.solve(gy_matrix, construct_b(x,y,t,params))
        
        @jax.jit
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

        # print('fsolve {}', self.g(x0,y0,T[0],params))

        y0new, infodict, ier, mesg = fsolve(lambda yy : self.g(x0, yy, T[0], params), y0, full_output=True)
        
        if ier != 1:
            # throw an error if the algebraic states are not consistent.
            raise ValueError("Initial algebraic states were inconsistent. fsolve returned {}".format(mesg))

        if not (jnp.abs(y0new - y0) < y0_tol).all():
            # print("Initial algebraic states {} were inconsistent. New initial algebraic state values are {}".format(y0, y0new))
            y0 = y0new
            z0 = jnp.concatenate((x0, y0))

        sol = odeint(self.f_coupled_system, z0, T, params)

        return sol
    
    def get_consistent_initial_condition(self, z0, t0, params, y0_tol=1e-9):
        x0 = z0[0:self.num_diff_vars]
        y0 = z0[self.num_diff_vars::]

        y0new, infodict, ier, mesg = fsolve(lambda yy : self.g(x0, yy, t0, params), y0, full_output=True)

        if ier != 1:
            # throw an error if the algebraic states are not consistent.
            raise ValueError("Initial algebraic states were inconsistent. fsolve returned {}".format(mesg))

        if not (jnp.abs(y0new - y0) < y0_tol).all():
            print("Initial algebraic states {} were inconsistent. New initial algebraic state values are {}".format(y0, y0new))
            y0 = y0new
            z0 = jnp.concatenate((x0, y0))
        
        return z0
    
    def solve_dae_one_timestep_rk4(self, z0, t0, delta_t, params):
        """
        Solve one forward timestep using RK4.
        Warning that this method does not check for consistency of the algebraic states.
        """
        k1 = self.f_coupled_system(z0, t0, params)
        k2 = self.f_coupled_system(z0 + delta_t/2 * k1, t0 + delta_t/2, params)
        k3 = self.f_coupled_system(z0 + delta_t/2 * k2, t0 + delta_t/2, params)
        k4 = self.f_coupled_system(z0 + delta_t * k3, t0 + delta_t, params)

        return z0 + delta_t/6 * (k1 + 2 * k2 + 2 * k3 + k4)
    
    def solve_dae_one_timestep_implicit_euler(self, z0, t0, delta_t, params):
        """
        Solve one forward timestep of the implicit backwards Euler rule using Newton's method
        Warning that this method does not check for consistency of the algebraic states
        """
        f = lambda z1, params: z0 + 0.5 * delta_t * self.f_coupled_system(z1, t0+delta_t, params)
        return fixed_point_layer(fwd_solver, f, z0, params)
    
    def solve_dae_one_timestep_implicit_trapezoid(self, z0, t0, delta_t, params):
        """
        Solve one forward timestep of the implicit trapezoid rule using Newton's method
        Warning that this method does not check for consistency of the algebraic states
        """
        f = lambda z1, params: z0 + 0.5 * delta_t * (self.f_coupled_system(z0, t0, params) + self.f_coupled_system(z1, t0+delta_t, params))
        return fixed_point_layer(newton_solver, f, z0, params)


    def get_one_timestep_solver(self, name):

        one_timestep_solver_factory = {
            'rk4': self.solve_dae_one_timestep_rk4,
            'implicit_euler' : self.solve_dae_one_timestep_implicit_euler,
            'implicit_trapezoid' : self.solve_dae_one_timestep_implicit_trapezoid,
        }

        return one_timestep_solver_factory[name]

def main():

    def f(x, y, t, params):
        return -x + y

    def g(x, y, t, params):
        return x + y - 1

    solver = DAESolver(f, g, num_diff_vars=1, num_alg_vars=1, regularization_method='none', reg_param=0.0, one_timestep_solver='implicit_euler')

    diff_vars_init = jnp.array([0.0])
    alg_vars_init = jnp.array([1.0])

    z0 = jnp.concatenate((diff_vars_init, alg_vars_init))
    T = jnp.linspace(0, 1, 100)

    dt = 1e-2
    next_z = z0
    sol = []
    for t in range(100):
        t0 = t*dt
        next_z = solver.get_consistent_initial_condition(next_z, t0, params=None)
        next_z = solver.one_timestep_solver(next_z, t0, dt, params=None)
        sol.append(next_z)
    sol = jnp.array(sol)

    print(sol)

    sol_explicit = solver.solve_dae(z0, T, params=None)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sol[:,0] + sol[:,1])
    ax.plot(sol_explicit[:,0] + sol_explicit[:,1])
    plt.show()

if __name__ == "__main__":
    main()