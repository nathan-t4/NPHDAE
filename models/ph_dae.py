import jax.numpy as jnp
import jax
import sys
sys.path.append('../')
from dae_solver.index1_semi_explicit import DAESolver
import matplotlib.pyplot as plt

class PHDAE():

    def __init__(
            self, 
            AC : jnp.ndarray, 
            AR : jnp.ndarray, 
            AL : jnp.ndarray, 
            AV : jnp.ndarray,
            AI : jnp.ndarray,
            grad_H_func : callable, # Hamiltonian of inductors as a function of flux
            q_func : callable, # Charge as a function of voltage delta
            r_func : callable, # Current through resistor as a function of voltage delta
            u_func : callable, # Vector-valued function of time of currents/voltages supplied by the independent sources
        ):

        assert AC.shape[0] == AR.shape[0] == AL.shape[0] == AV.shape[0] == AI.shape[0], "All matrices must have the same number of rows."

        self.num_nodes = AC.shape[0] # number of non-ground nodes
        self.num_capacitors = AC.shape[1]
        self.num_inductors = AL.shape[1]
        self.num_voltage_sources = AV.shape[1]
        self.num_current_sources = AI.shape[1]
        self.num_resistors = AR.shape[1]

        if (AC == 0.0).all():
            self.num_capacitors = 0
        if (AR == 0.0).all():
            self.num_resistors = 0
        if (AL == 0.0).all():
            self.num_inductors = 0
        if (AV == 0.0).all():
            self.num_voltage_sources = 0
        if (AI == 0.0).all():
            self.num_current_sources = 0

        # There is a differential variable for each capacitor and inductor in the circuit
        self.num_differential_vars = self.num_capacitors + self.num_inductors

        # There is an algebraic variable for the voltage at each node, and for the current through each voltage source
        self.num_algebraic_vars = self.num_nodes + self.num_voltage_sources 

        self.AC = AC
        self.AR = AR
        self.AL = AL
        self.AV = AV
        self.AI = AI

        self.grad_H_func = grad_H_func
        self.q_func = q_func
        self.r_func = r_func
        self.u_func = u_func

        self.construct_f_and_g()
        self.construct_dae_solver()

    def construct_f_and_g(self):

        # These don't dynamically change
        diffeq_indices, alg_eq_indices = self.get_diffeq_indexes_in_output_vector()

        def f(x, y, t, params):
            if params is not None:
                if 'u_func' in params.keys():
                    u_func_params = params['u_func']
                else:
                    u_func_params = None
            else:
                u_func_params = None

            E, J, z_vec, diss, B = self.construct_matrix_equations(x, y, t, params) 

            rhs = jnp.linalg.matmul(J, z_vec) - diss + jnp.linalg.matmul(B, self.u_func(t, u_func_params))

            # Grab only the rows corresponding to the differential equation indices and columns corresponding to the differential variables
            rhs = rhs[diffeq_indices]
            E = E[diffeq_indices][:, 0:(self.num_capacitors + self.num_inductors)]

            # Note that this system could actually be overdetermined.
            # It will always have a unique solution, but there could be more equations that unknowns.
            if E.shape[0] != E.shape[1]:
                # Truncate a QR factorization to deal with redundant equations.
                Q, R = jnp.linalg.qr(E) # 
                rhs = jnp.linalg.matmul(Q.transpose(), rhs) 
                E = R[0:(self.num_capacitors + self.num_inductors)]

            return jnp.linalg.solve(E, rhs)

        def g(x, y, t, params):
            if params is not None:
                if 'u_func' in params.keys():
                    u_func_params = params['u_func']
                else:
                    u_func_params = None
            else:
                u_func_params = None
        
            E, J, z_vec, diss, B = self.construct_matrix_equations(x, y, t, params)
            rhs = jnp.linalg.matmul(J, z_vec) - diss + jnp.linalg.matmul(B, self.u_func(t, u_func_params))

            return rhs[alg_eq_indices]

        self.f = jax.jit(f)
        self.g = jax.jit(g)

    def construct_E_matrix(self):
        return jax.scipy.linalg.block_diag(self.AC, jnp.eye(self.num_inductors), jnp.zeros((self.num_capacitors, self.num_nodes)), jnp.zeros(self.num_voltage_sources))
    
    def get_diffeq_indexes_in_output_vector(self):
        """
        The DAE is expressed as a large system of equations.
        This function returns the indecies (rows) of this system of equations that 
        corresponds to differential equations.
        """
        E = self.construct_E_matrix()

        # Check whether the rows of the E matrix are nonzero.
        diffeq_indices = jnp.where(jnp.array([(E[row, :] != 0.0).any() for row in range(E.shape[0])]))
        alg_eq_indices = jnp.where(jnp.array([(E[row, :] == 0.0).all() for row in range(E.shape[0])]))

        return diffeq_indices, alg_eq_indices

    def construct_matrix_equations(self, x, y, t, params):
        q = x[0:self.num_capacitors]
        phi = x[self.num_capacitors::]

        e = y[0:self.num_nodes]
        jv = y[self.num_nodes::]

        # Parse the parameters passed in to parametrize the various PH_DAE function evaluations
        if params is not None:
            if 'r_func' in params.keys():
                r_func_params = params['r_func']
            else:
                r_func_params = None
            if 'grad_H_func' in params.keys():
                grad_H_func_params = params['grad_H_func']
            else:
                grad_H_func_params = None
            if 'q_func' in params.keys():
                q_func_params = params['q_func']
            else:
                q_func_params = None
        else:
            r_func_params = None
            grad_H_func_params = None
            q_func_params = None

        output_dim = self.num_nodes + self.num_inductors + self.num_capacitors + self.num_voltage_sources

        E = self.construct_E_matrix()

        # Construct the J matrix
        J = jnp.zeros((output_dim, self.num_differential_vars + self.num_algebraic_vars))

        # top equations of J
        J = J.at[0:self.num_nodes, self.num_nodes:(self.num_nodes + self.num_inductors)].set(-self.AL)
        J = J.at[
            0:(self.num_nodes),
            (self.num_nodes + self.num_inductors + self.num_capacitors)::
            ].set(-self.AV)
    
        # second row of equations of J
        J = J.at[self.num_nodes:(self.num_nodes + self.num_inductors),
                0:self.num_nodes].set(self.AL.transpose())
        
        # Final row of equations of J
        J = J.at[(self.num_nodes + self.num_inductors + self.num_capacitors)::,
                0:self.num_nodes].set(self.AV.transpose())
        
        # Construct the dissipation term
        diss = jnp.zeros((output_dim,))
        curr_through_resistors = jnp.linalg.matmul(self.AR, self.r_func(jnp.linalg.matmul(self.AR.transpose(), e), r_func_params)) # The current through the resistors.
        diss = diss.at[0:self.num_nodes].set(curr_through_resistors)

        charge_constraint = self.q_func(jnp.linalg.matmul(self.AC.transpose(), e), q_func_params) - q
        diss = diss.at[(self.num_nodes + self.num_inductors):(self.num_nodes + self.num_inductors + self.num_capacitors)].set(charge_constraint)

        # Construct the input matrix B
        B = jnp.zeros((output_dim, self.num_current_sources + self.num_voltage_sources))
        B = B.at[0:self.num_nodes, 0:self.num_current_sources].set(-self.AI)
        B = B.at[(self.num_nodes + self.num_inductors + self.num_capacitors)::, self.num_current_sources::].set(-jnp.eye(self.num_voltage_sources))

        # Construct z_vec
        z_vec = jnp.concatenate((e, self.grad_H_func(phi, grad_H_func_params), q, jv))
    
        return E, J, z_vec, diss, B

    def construct_dae_solver(self):
        self.solver = DAESolver(self.f, self.g, self.num_differential_vars, self.num_differential_vars)

    def solve(self, z0, T, params, tol=1e-6):
        return self.solver.solve_dae(z0, T, params, tol)
    
if __name__ == "__main__":
    AC = jnp.array([[0.0], [0.0], [1.0]])
    AR = jnp.array([[1.0], [-1.0], [0.0]])
    AL = jnp.array([[0.0], [1.0], [-1.0]])
    AV = jnp.array([[1.0], [0.0], [0.0]])
    AI = jnp.array([[0.0], [0.0], [0.0]])

    R = 1
    L = 1
    C = 1

    x0 = jnp.array([0.0, 0.0])
    y0 = jnp.array([0.0, 0.0, 0.0, 0.0])
    z0 = jnp.concatenate((x0, y0))
    T = jnp.linspace(0, 1.5, 1000)

    def r_func(delta_V, params=None):
        return delta_V / R
    
    def q_func(delta_V, params=None):
        return C * delta_V
    
    def grad_H_func(phi, params=None):
        return phi / L
    
    def u_func(t, params):
        return jnp.array([jnp.sin(30 * t)])
    
    dae = PHDAE(AC, AR, AL, AV, AI, grad_H_func=grad_H_func, q_func=q_func, r_func=r_func, u_func=u_func)

    sol = dae.solve(z0, T, params=None)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sol[:,5])
    plt.show()

    print(sol)