import jax.numpy as jnp
import jax
import sys
sys.path.append('../')
from dae_solver.index1_semi_explicit import DAESolver
import matplotlib.pyplot as plt

class CompositePHDAE():

    def __init__(
            self, 
            ph_dae_list : list,
            Alambda : jnp.ndarray,
            regularization_method : str = 'none',
            reg_param : float = 0.0,
            one_timestep_solver : str = 'rk4',
        ):

        self.num_subsystems = len(ph_dae_list)
        self.regularization_method = regularization_method
        self.reg_param = reg_param
        self.one_timestep_solver = one_timestep_solver
        
        self.num_nodes = sum([dae.num_nodes for dae in ph_dae_list])
        self.num_capacitors = sum([dae.num_capacitors for dae in ph_dae_list])
        self.num_inductors = sum([dae.num_inductors for dae in ph_dae_list])
        self.num_voltage_sources = sum([dae.num_voltage_sources for dae in ph_dae_list])
        self.num_current_sources = sum([dae.num_current_sources for dae in ph_dae_list])
        self.num_resistors = sum([dae.num_resistors for dae in ph_dae_list])

        self.ph_dae_list = ph_dae_list
        self.Alambda = Alambda

        assert Alambda.shape[0] == self.num_nodes, "The Alambda matrix must have the same number of rows as the number of nodes in the composite PHDAE."
        self.num_couplings = Alambda.shape[1]

        # There is a differential variable for each capacitor and inductor in the circuit
        self.num_differential_vars = self.num_capacitors + self.num_inductors

        # There is an algebraic variable for the voltage at each node, and for the current through each voltage source and the current through each coupling
        self.num_algebraic_vars = self.num_nodes + self.num_voltage_sources + self.num_couplings

        # Build matrix AC
        AC = jnp.zeros((self.num_nodes, self.num_capacitors))
        last_row_ind = 0
        last_col_ind = 0
        for dae in self.ph_dae_list:
            if dae.num_capacitors > 0:
                AC = AC.at[last_row_ind:(last_row_ind + dae.num_nodes), last_col_ind:(last_col_ind + dae.num_capacitors)].set(dae.AC)
                last_row_ind += dae.num_nodes
                last_col_ind += dae.num_capacitors
            else:
                last_row_ind += dae.num_nodes
        self.AC = AC

        # Build matrix AR
        AR = jnp.zeros((self.num_nodes, self.num_resistors))
        last_row_ind = 0
        last_col_ind = 0
        for dae in self.ph_dae_list:
            if dae.num_resistors > 0:
                AR = AR.at[last_row_ind:(last_row_ind + dae.num_nodes), last_col_ind:(last_col_ind + dae.num_resistors)].set(dae.AR)
                last_row_ind += dae.num_nodes
                last_col_ind += dae.num_resistors
            else:
                last_row_ind += dae.num_nodes
        self.AR = AR

        # Build matrix AL
        AL = jnp.zeros((self.num_nodes, self.num_inductors))
        last_row_ind = 0
        last_col_ind = 0
        for dae in self.ph_dae_list:
            if dae.num_inductors > 0:
                AL = AL.at[last_row_ind:(last_row_ind + dae.num_nodes), last_col_ind:(last_col_ind + dae.num_inductors)].set(dae.AL)
                last_row_ind += dae.num_nodes
                last_col_ind += dae.num_inductors
            else:
                last_row_ind += dae.num_nodes
        self.AL = AL

        # Build matrix AV
        AV = jnp.zeros((self.num_nodes, self.num_voltage_sources))
        last_row_ind = 0
        last_col_ind = 0
        for dae in self.ph_dae_list:
            if dae.num_voltage_sources > 0:
                AV = AV.at[last_row_ind:(last_row_ind + dae.num_nodes), last_col_ind:(last_col_ind + dae.num_voltage_sources)].set(dae.AV)
                last_row_ind += dae.num_nodes
                last_col_ind += dae.num_voltage_sources
            else:
                last_row_ind += dae.num_nodes
        self.AV = AV

        # Build matrix AI
        AI = jnp.zeros((self.num_nodes, self.num_current_sources))
        last_row_ind = 0
        last_col_ind = 0
        for dae in self.ph_dae_list:
            if dae.num_current_sources > 0:
                AI = AI.at[last_row_ind:(last_row_ind + dae.num_nodes), last_col_ind:(last_col_ind + dae.num_current_sources)].set(dae.AI)
                last_row_ind += dae.num_nodes
                last_col_ind += dae.num_current_sources
            else:
                last_row_ind += dae.num_nodes
        self.AI = AI

        self.construct_f_and_g()
        self.construct_dae_solver()

    def construct_f_and_g(self):

        # These don't dynamically change
        diffeq_indices, alg_eq_indices = self.get_diffeq_indexes_in_output_vector()

        def f(x, y, t, params_list):
            E, J, z_vec, diss, B = self.construct_matrix_equations(x, y, t, params_list)

            u_output_list = []
            for subsystem_ind in range(self.num_subsystems):
                num_subsystem_current_sources = self.ph_dae_list[subsystem_ind].num_current_sources
                if num_subsystem_current_sources > 0:
                    _, _, _, u_func_params_of_subsystem = self.extract_params(params_list[subsystem_ind])
                    current_u_output_of_subsystem = self.ph_dae_list[subsystem_ind].u_func(t, u_func_params_of_subsystem)[0:num_subsystem_current_sources]
                    u_output_list.append(current_u_output_of_subsystem)
            for subsystem_ind in range(self.num_subsystems):
                num_subsystem_voltage_sources = self.ph_dae_list[subsystem_ind].num_voltage_sources
                num_subsystem_current_sources = self.ph_dae_list[subsystem_ind].num_current_sources
                if num_subsystem_voltage_sources > 0:
                    _, _, _, u_func_params_of_subsystem = self.extract_params(params_list[subsystem_ind])
                    voltage_u_output_of_subsystem = self.ph_dae_list[subsystem_ind].u_func(t, u_func_params_of_subsystem)[num_subsystem_current_sources::]
                    u_output_list.append(voltage_u_output_of_subsystem)
            u_output = jnp.concatenate(u_output_list)

            rhs = jnp.linalg.matmul(J, z_vec) - diss + jnp.linalg.matmul(B, u_output)

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

        def g(x, y, t, params_list):
        
            E, J, z_vec, diss, B = self.construct_matrix_equations(x, y, t, params_list)

            u_output_list = []
            # last_idx = 0
            for subsystem_ind in range(self.num_subsystems):
                num_subsystem_current_sources = self.ph_dae_list[subsystem_ind].num_current_sources
                if num_subsystem_current_sources > 0:
                    _, _, _, u_func_params_of_subsystem = self.extract_params(params_list[subsystem_ind])
                    # u_func_params_of_subystem = params_list['composite_u_func'][last_idx : last_idx+num_subsystem_current_sources]
                    # last_idx += num_subsystem_current_sources
                    current_u_output_of_subsystem = self.ph_dae_list[subsystem_ind].u_func(t, u_func_params_of_subsystem)[0:num_subsystem_current_sources]
                    u_output_list.append(current_u_output_of_subsystem)
            # last_idx = 0
            for subsystem_ind in range(self.num_subsystems):
                num_subsystem_voltage_sources = self.ph_dae_list[subsystem_ind].num_voltage_sources
                num_subsystem_current_sources = self.ph_dae_list[subsystem_ind].num_current_sources
                if num_subsystem_voltage_sources > 0:
                    _, _, _, u_func_params_of_subsystem = self.extract_params(params_list[subsystem_ind])
                    # u_func_params_of_subystem = params_list['composite_u_func'][last_idx : last_idx+num_subsystem_current_sources]
                    # last_idx += num_subsystem_current_sources
                    voltage_u_output_of_subsystem = self.ph_dae_list[subsystem_ind].u_func(t, u_func_params_of_subsystem)[num_subsystem_current_sources::]
                    u_output_list.append(voltage_u_output_of_subsystem)

            u_output = jnp.concatenate(u_output_list)

            rhs = jnp.linalg.matmul(J, z_vec) - diss + jnp.linalg.matmul(B, u_output)

            return rhs[alg_eq_indices]

        self.f = jax.jit(f)
        self.g = jax.jit(g)

    def construct_E_matrix(self):
        E = jnp.zeros((self.num_nodes + self.num_inductors + self.num_capacitors + self.num_voltage_sources + self.num_couplings, self.num_nodes + self.num_inductors + self.num_capacitors + self.num_voltage_sources + self.num_couplings))
        # return jax.scipy.linalg.block_diag(self.AC, jnp.eye(self.num_inductors), jnp.zeros((self.num_capacitors, self.num_nodes)), jnp.zeros(self.num_voltage_sources), jnp.zeros(self.num_couplings))
        E = E.at[0:self.num_nodes, 0:self.num_capacitors].set(self.AC)
        E = E.at[self.num_nodes:(self.num_nodes + self.num_inductors), self.num_capacitors:(self.num_capacitors + self.num_inductors)].set(jnp.eye(self.num_inductors))
        return E

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

    def extract_params(self, params_dict):
        if params_dict is not None:
            if 'r_func' in params_dict.keys():
                r_func_params = params_dict['r_func']
            else:
                r_func_params = None
            if 'grad_H_func' in params_dict.keys():
                grad_H_func_params = params_dict['grad_H_func']
            else:
                grad_H_func_params = None
            if 'q_func' in params_dict.keys():
                q_func_params = params_dict['q_func']
            else:
                q_func_params = None
            if 'u_func' in params_dict.keys():
                u_func_params = params_dict['u_func']
            else:
                u_func_params = None
        else:
            r_func_params = None
            grad_H_func_params = None
            q_func_params = None
            u_func_params = None

        return r_func_params, grad_H_func_params, q_func_params, u_func_params

    def construct_matrix_equations(self, x, y, t, params_list):
        q = x[0:self.num_capacitors]
        phi = x[self.num_capacitors::]

        e = y[0:self.num_nodes]
        jv = y[self.num_nodes:(self.num_nodes + self.num_voltage_sources)]
        lamb = y[(self.num_nodes + self.num_voltage_sources)::]

        output_dim = self.num_nodes + self.num_inductors + self.num_capacitors + self.num_voltage_sources + self.num_couplings

        E = self.construct_E_matrix()

        # Construct the J matrix
        J = jnp.zeros((output_dim, self.num_differential_vars + self.num_algebraic_vars))

        # top equations of J
        J = J.at[0:self.num_nodes, self.num_nodes:(self.num_nodes + self.num_inductors)].set(-self.AL)
        J = J.at[
            0:(self.num_nodes),
            (self.num_nodes + self.num_inductors + self.num_capacitors):(self.num_nodes + self.num_inductors + self.num_capacitors + self.num_voltage_sources)
            ].set(-self.AV)
        J = J.at[
            0:(self.num_nodes),
            (self.num_nodes + self.num_inductors + self.num_capacitors + self.num_voltage_sources)::
            ].set(-self.Alambda)
    
        # second row of equations of J
        J = J.at[self.num_nodes:(self.num_nodes + self.num_inductors),
                0:self.num_nodes].set(self.AL.transpose())
        
        # Voltage row equations of J
        J = J.at[(self.num_nodes + self.num_inductors + self.num_capacitors):(self.num_nodes + self.num_inductors + self.num_capacitors + self.num_voltage_sources),
                0:self.num_nodes].set(self.AV.transpose())
        
        # Coupling row equations of J
        J = J.at[(self.num_nodes + self.num_inductors + self.num_capacitors + self.num_voltage_sources)::,
                0:self.num_nodes].set(self.Alambda.transpose())
        
        # Construct the dissipation term
        diss = jnp.zeros((output_dim,))

        delta_v_resistors = jnp.linalg.matmul(self.AR.transpose(), e)
        r_func_outputs = []
        last_start_ind = 0
        for subsystem_ind in range(self.num_subsystems):
            if self.ph_dae_list[subsystem_ind].num_resistors > 0:
                delta_v_of_subsystem = delta_v_resistors[last_start_ind:(last_start_ind + self.ph_dae_list[subsystem_ind].num_resistors)]
                last_start_ind += self.ph_dae_list[subsystem_ind].num_resistors
                r_func_params_of_subsystem, _, _, _ = self.extract_params(params_list[subsystem_ind])
                r_func_of_subsystem = self.ph_dae_list[subsystem_ind].r_func(delta_v_of_subsystem, r_func_params_of_subsystem)
                r_func_outputs.append(r_func_of_subsystem)
        r_func_outputs = jnp.concatenate(r_func_outputs)

        curr_through_resistors = jnp.linalg.matmul(self.AR, r_func_outputs) # The current through the resistors.
        diss = diss.at[0:self.num_nodes].set(curr_through_resistors)

        delta_v_capacitors = jnp.linalg.matmul(self.AC.transpose(), e)
        q_func_outputs = []
        last_start_ind = 0
        for subsystem_ind in range(self.num_subsystems):
            if self.ph_dae_list[subsystem_ind].num_capacitors:
                delta_v_of_subsystem = delta_v_capacitors[last_start_ind:(last_start_ind + self.ph_dae_list[subsystem_ind].num_capacitors)]
                last_start_ind += self.ph_dae_list[subsystem_ind].num_capacitors
                _, _, q_func_params_of_subsystem, _ = self.extract_params(params_list[subsystem_ind])
                q_func_of_subsystem = self.ph_dae_list[subsystem_ind].q_func(delta_v_of_subsystem, q_func_params_of_subsystem)
                q_func_outputs.append(q_func_of_subsystem)
        q_func_outputs = jnp.concatenate(q_func_outputs)

        charge_constraint = q_func_outputs - q
        diss = diss.at[(self.num_nodes + self.num_inductors):(self.num_nodes + self.num_inductors + self.num_capacitors)].set(charge_constraint)

        # Construct the input matrix B
        B = jnp.zeros((output_dim, self.num_current_sources + self.num_voltage_sources))
        B = B.at[0:self.num_nodes, 0:self.num_current_sources].set(-self.AI)
        B = B.at[(self.num_nodes + self.num_inductors + self.num_capacitors):(self.num_nodes + self.num_inductors + self.num_capacitors + self.num_voltage_sources), self.num_current_sources::].set(-jnp.eye(self.num_voltage_sources))

        # Construct z_vec
        grad_H_outputs = []
        last_ind = 0
        for subsystem_ind in range(self.num_subsystems):
            if self.ph_dae_list[subsystem_ind].num_inductors > 0:
                phi_of_subsystem = phi[last_ind:(last_ind + self.ph_dae_list[subsystem_ind].num_inductors)]
                last_ind += self.ph_dae_list[subsystem_ind].num_inductors
                _, grad_H_params_of_subsystem, _, _ = self.extract_params(params_list[subsystem_ind])
                grad_H_outputs_of_subsystem = self.ph_dae_list[subsystem_ind].grad_H_func(phi_of_subsystem, grad_H_params_of_subsystem)
                grad_H_outputs.append(grad_H_outputs_of_subsystem)
        grad_H_outputs = jnp.concatenate(grad_H_outputs)

        z_vec = jnp.concatenate((e, grad_H_outputs, q, jv, lamb))
    
        return E, J, z_vec, diss, B

    def construct_dae_solver(self):
        self.solver = DAESolver(self.f, self.g, self.num_differential_vars, self.num_algebraic_vars, self.one_timestep_solver)

    def solve(self, z0, T, params_list, tol=1e-6, is_training=False, control=None):
        """ Use the semi-explicit DAE solver to compute the trajectory """
        if not is_training:
            params_list = jax.lax.stop_gradient(params_list)
        # if control is not None:
        #     params_list['composite_u_func'] = control
        return self.solver.solve_dae(z0, T, params_list, tol)
        
    def solve_one_timestep(self, z0, T, params_list, tol=1e-6, consistent_ic=False, is_training=False, control=None):
        """ Use the one timestep solver to compute the trajectory """
        # if control is not None:
        #     params_list['composite_u_func'] = control

        # Disable gradients if not training
        if not is_training:
            params_list = jax.lax.stop_gradient(params_list)
        
        zs = []
        z = self.solver.get_consistent_initial_condition(z0, T[0], params_list, tol) if not consistent_ic else z0
        print("Done solving initial condition")
        dt = T[1] - T[0]
        for i in range(len(T)):
            dt = T[i+1] - T[i] if i != len(T) - 1 else dt
            if consistent_ic:
                z = self.solver.get_consistent_initial_condition(z, T[i], params_list)
            zs.append(z)
            z = self.solver.one_timestep_solver(z, T[i], dt, params_list)
        
        return jnp.array(zs)