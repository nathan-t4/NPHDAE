import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def newton_raphson(fun, jac, x0, tol=1e-6, max_iter=50):
    """
    Newton-Raphson solver for finding the roots of a nonlinear system.

    Parameters:
    fun (callable): Function representing the system of equations.
    jac (callable): Function to compute the Jacobian matrix of the system.
    x0 (array): Initial guess for the solution.
    tol (float): Tolerance for convergence.
    max_iter (int): Maximum number of iterations.

    Returns:
    array: Solution vector.
    """
    x = x0
    for i in range(max_iter):
        f_val = fun(x)
        j_val = jac(x)
        delta_x = jnp.linalg.solve(j_val, -f_val)
        x = x + delta_x

        if jnp.linalg.norm(delta_x) < tol:
            break

    return x

class RadauIIA:

    def __init__(self, fun):
        self.fun = fun

        # Radau IIA coefficients for s=2 stages (3rd-order method)
        self.A = jnp.array([
            [5/12, -1/12],
            [3/4, 1/4]
        ])
        self.b = jnp.array([3/4, 1/4])
        self.c = jnp.array([1/3, 1])

        self.num_stages = 2

    def solve_iterate_through_time(self, y0, yp0, tspan, h):
        t = jnp.arange(tspan[0], tspan[1], h)
        y = jnp.zeros((len(t), len(y0)))
        yp = jnp.zeros((len(t), len(yp0)))
        y = y.at[0, :].set(y0)
        yp = yp.at[0, :].set(yp0)

        state_dim = len(y0)

        for i in range(1, len(t)):
            t_n = t[i-1]
            y_n = y[i-1]
            yp_n = yp[i-1]

            res_fn = self.construct_residuals_single_timestep(y_n, yp_n, t_n, h, state_dim)
            jac_fn = jax.jit(jax.jacfwd(res_fn))

            # Initial guess for stage values and their derivatives
            Y_guess = jnp.tile(y_n, 2)
            Yp_guess = jnp.tile(yp_n, 2)

            # Concatenate the stage values and their derivatives
            x_guess = jnp.hstack((Y_guess, Yp_guess))

            # Solve the system of nonlinear equations
            x = newton_raphson(res_fn, jac_fn, x_guess)

            # Extract the solution stage values and their derivatives
            Y = x[:state_dim*self.num_stages].reshape(state_dim, self.num_stages)
            Yp = x[state_dim*self.num_stages:].reshape(state_dim, self.num_stages)

            # Update the solution at the current time step
            y = y.at[i, :].set(
                y_n + h * jnp.sum(
                    jnp.array([self.b[j] * Yp[:,j] for j in range(self.num_stages)])
                )
            )

            y_next = newton_raphson(
                lambda x: self.fun(t[i], y[i, :], x), 
                jax.jacfwd(lambda x: self.fun(t[i], y[i, :], x)), 
                yp_n
            )

            yp = yp.at[i, :].set(
                y_next
            )

        return t, y, yp

    def construct_residuals_single_timestep(self, y, yp, t, h, state_dim):

        assert y.shape == (state_dim,)
        assert yp.shape == (state_dim,)

        def res_fn(x):
            Y = x[:state_dim*self.num_stages].reshape(self.num_stages, state_dim)
            Yp = x[state_dim*self.num_stages:].reshape(self.num_stages, state_dim)

            equation_residuals = []

            for j in range(self.num_stages):
                equation_residuals.append(
                    Y[j, :] - y - h * jnp.sum(jnp.array([self.A[j, k] * Yp[k, :] for k in range(self.num_stages)]))
                )
                equation_residuals.append(
                    self.fun(t + self.c[j]*h, Y[j, :], Yp[j, :])
                )

            return jnp.hstack(equation_residuals)

        return res_fn

# def radau_iia_2(fun, t_span, y0, yp0, h):
#     t = jnp.arange(t_span[0], t_span[1], h)
#     y = jnp.zeros((len(t), len(y0)))
#     yp = jnp.zeros((len(t), len(yp0)))
#     y = y.at[0].set(y0)
#     yp = yp.at[0].set(yp0)

#     # Radau IIA coefficients for s=2 stages (3rd-order method)
#     A = jnp.array([
#         [5/12, -1/12],
#         [3/4, 1/4]
#     ])
#     b = jnp.array([3/4, 1/4])
#     c = jnp.array([1/3, 1])

#     num_stages = 2

#     def construct_residuals_entire_time_horizon(Y, Yp, y, yp, y0, yp0, h):
#         equation_residuals = []

#         y = y.at[0].set(y0)
#         yp = yp.at[0].set(yp0)

#         for i in range(len(t)):
#             # stage equations
#             for j in range(num_stages):
#                 equation_residuals.append(
#                     Y[i, :, j] - y[i] - h * jnp.sum(A[j, k] * Yp[i, :, k] for k in range(num_stages))
#                 )
#                 equation_residuals.append(
#                     fun(t[i] + c[j]*h, Y[i, :, j], Yp[i, :, j])
#                 )

#             equation_residuals.append(
#                 y[i] + h * jnp.sum(b[j] * Yp[i, :, j] for j in range(num_stages))
#             )

#         return jnp.asarray(equation_residuals)

#     # Intermediate function to be able to pass in first two arguments as autodiff.
#     def res_fn(input):
#         Y, Yp, y, yp = input
#         return construct_residuals_entire_time_horizon(Y, Yp, y, yp, y0, yp0, h)

#     jacobian = jax.jit(jax.jacfwd(res_fn))

#     Y_guess = jnp.tile(jnp.zeros(len(t)), y0, num_stages)
#     Yp_guess = jnp.tile(jnp.zeros(len(t)), yp0, num_stages)
#     # for each stage, we have a guess for the state and the derivative
#     # Because no better information is available, we can use the initial guess
#     y_guess = jnp.tile(y0, num_stages)
#     yp_guess = jnp.tile(yp0, num_stages)

#     x_guess = (Y_guess, Yp_guess, y_guess, yp_guess)

#     x = newton_raphson(res_fn, jacobian, x_guess)

# Example usage
if __name__ == "__main__":
    # def f(t, y, yp):
    #     return jnp.array([yp[0] - y[1], y[0]**2 + y[1]**2 - 1])

    # t_span = [0, 1]
    # y0 = jnp.array([1, 0])
    # yp0 = jnp.array([0, 1])
    # h = 0.01

    def f(t, y, yp, params=jnp.array([0.1, 0.01])):
        """Define implicit system of differential algebraic equations."""

        # Example RC circuit from this page: https://en.wikipedia.org/wiki/Modified_nodal_analysis#:~:text=In%20electrical%20engineering%2C%20modified%20nodal,but%20also%20some%20branch%20currents.

        C, G = params

        A = jnp.array([
            [G, -G, 1,],
            [-G, G, 0],
            [1, 0, 0]
        ])

        E = jnp.array([
            [0, 0, 0],
            [0, C, 0],
            [0, 0, 0]
        ])

        f = jnp.array([0, 0, jnp.sin(3 * t)])

        F = E @ yp + A @ y - f
        return F

    # time span
    t0 = 0
    t1 = 1.5
    t_span = (t0, t1)

    # initial conditions
    y0 = jnp.array([0, 0, 0], dtype=float)
    yp0 = jnp.array([0, 0, 0], dtype=float)

    radau_iia = RadauIIA(f)
    t, y, yp = radau_iia.solve_iterate_through_time(y0, yp0, t_span, 0.01)

    # t, y = radau_iia_2(fun, t_span, y0, yp0, h)

    plt.plot(t, y[:, 0], label='y1')
    plt.plot(t, y[:, 1], label='y2')
    plt.legend()
    plt.xlabel('Time t')
    plt.ylabel('Solution y')
    plt.title('Radau IIA Method for Index-2 DAE')
    plt.show()
