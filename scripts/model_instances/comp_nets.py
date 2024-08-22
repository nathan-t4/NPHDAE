import jraph
import jax
import flax.linen as nn
import jax.numpy as jnp
from typing import Callable, Sequence
from flax.training.train_state import TrainState
from flax.typing import Array
from helpers.integrator_factory import integrator_factory
from utils.graph_utils import *
from utils.jax_utils import *
from utils.models_utils import *
from utils.gnn_utils import *
from utils.comp_utils import *
from dae_solver.index1_semi_explicit_flax import DAESolver
from scipy.optimize import fsolve, minimize
import time


class CompLCGNSOld(nn.Module):
    integration_method: str
    dt: float
    T: int
    state_one: TrainState
    state_two: TrainState 
    graph_from_state_one: Callable = None # TODO
    graph_from_state_two: Callable = None # TODO

    @nn.compact
    def __call__(self, graph1, graph2, rng):
        senders1 = graph1.senders
        receivers1 = graph1.receivers
        senders2 = graph2.senders
        receivers2 = graph2.receivers

        cur_nodes1 = graph1.nodes
        state1 = graph1.edges[:,0].squeeze()
        Q1, Phi1, Q3_1 = state1

        cur_nodes2 = graph2.nodes
        state2 = graph2.edges[:,0].squeeze()
        Q2, Phi2, Q3_2 = state2
        full_state = jnp.concatenate((state1, state2[:2]))
        # full_state = jnp.concatenate((state1, state2))
        control1 = jnp.array([0, 0, 0])
        # control2 = jnp.array([0, V3])
        control2 = jnp.array([0, 0, 0])

        # J1 = self.state_one.params['params']['Dense_0']['kernel']
        # J2 = self.state_two.params['params']['Dense_0']['kernel']

        # J1 = jnp.triu(J1) - jnp.triu(J1).T
        # J2 = jnp.triu(J2) - jnp.triu(J2).T

        J1 = jnp.array([[0, 1, 0],
                        [-1, 0, 1],
                        [0, -1, 0]])
        J2 = jnp.array([[0, 1],
                        [-1, 0]])
        C = jnp.array([[0, 0],
                       [0, 0],
                       [0, -1]])
        
        Jc = jnp.block([[J1, C],
                        [-C.T, J2]])
                
        def H_from_state(x):
            # Modify node voltages and edges to satisfy Kirchhoff's laws
            Q1, Phi1, Q3, Q2, Phi2 = x
            edges1 = jnp.array([[Q1, 0],
                                [Phi1, 1],
                                [Q3, 0]])
            globals1 = None
            n_node1 = jnp.array([len(cur_nodes1)])
            n_edge1 = jnp.array([len(edges1)])

            edges2 = jnp.array([[Q2, 0],
                                [Phi2, 1],
                                [Q3, 0]])
            globals2 = None
            n_node2 = jnp.array([len(cur_nodes2)])
            n_edge2 = jnp.array([len(edges2)])

            graph1 = jraph.GraphsTuple(nodes=cur_nodes1,
                                       edges=edges1,
                                       globals=globals1,
                                       senders=senders1, 
                                       receivers=receivers1,
                                       n_node=n_node1,
                                       n_edge=n_edge1)
            
            graph2 = jraph.GraphsTuple(nodes=cur_nodes2,
                                       edges=edges2,
                                       globals=globals2,
                                       senders=senders2, 
                                       receivers=receivers2,
                                       n_node=n_node2,
                                       n_edge=n_edge2)

            # graph1 = self.graph_from_state_one(state=x, control=control1, system_params=False, set_nodes=False, set_ground_and_control=False, nodes=cur_nodes1, globals=globals1)
            # graph2 = self.graph_from_state_two(state=x, control=control2, system_params=False, set_nodes=False, set_ground_and_control=False, nodes=cur_nodes2, globals=globals2)

            next_graph1 = self.state_one.apply_fn(self.state_one.params, graph1, control1, rng)
            next_graph2 = self.state_two.apply_fn(self.state_two.params, graph2, control2, rng)

            H1 = next_graph1.globals.squeeze()
            H2 = next_graph2.globals.squeeze()

            H = H1 + H2

            return H, (next_graph1, next_graph2)
        
        def dynamics_function(x, t, aux_data):
            dH, _ = jax.grad(H_from_state, has_aux=True)(x)           
            z = dH            
            return jnp.matmul(Jc, z).squeeze()
        
        H, (next_graph1, next_graph2) = H_from_state(full_state)
        aux_data = None
        # Integrate port-Hamiltonian dynamics
        next_state = None
        if self.integration_method == 'adam_bashforth':
            next_state = integrator_factory(self.integration_method)(partial(dynamics_function, aux_data=aux_data), full_state, 0, self.dt, self.T) 
        else:
            next_state = integrator_factory(self.integration_method)(partial(dynamics_function, aux_data=aux_data), full_state, 0, self.dt)
        
        next_Q1, next_Phi1, next_Q3, next_Q2, next_Phi2 = next_state
        # reset voltages to observed value...
        next_nodes1 = jnp.array([[0], [next_Q1], [next_Q3]]) # Assuming C = L = C_prime = 1 (params are known)
        next_edges1 = jnp.array([[next_Q1, 0],
                                 [next_Phi1, 1],
                                 [next_Q3, 0]])
        next_nodes2 = jnp.array([[0], [next_Q2], [next_Q3]])
        next_edges2 = jnp.array([[next_Q2, 0],
                                 [next_Phi2, 1],
                                 [next_Q3, 0]])
        next_graph1 = next_graph1._replace(edges=next_edges1,
                                           nodes=next_nodes1)
        next_graph2 = next_graph2._replace(edges=next_edges2,
                                           nodes=next_nodes2)

        # next_graph1 = self.graph_from_state(state=next_state1, 
        #                                     control=control1, 
        #                                     system_params=False, 
        #                                     set_nodes=True,
        #                                     set_ground_and_control=True, 
        #                                     nodes=next_nodes1, 
        #                                     globals=next_graph1.globals)
        
        # next_graph2 = self.graph_from_state(state=next_state2, 
        #                                     control=control1, 
        #                                     system_params=False, 
        #                                     set_nodes=True,
        #                                     set_ground_and_control=True, 
        #                                     nodes=next_nodes2, 
        #                                     globals=next_graph2.globals)
        return next_graph1, next_graph2
    
class CompPHGNS(nn.Module):
    ode_integration_method: str
    dt: float
    T: int
    train_states: Sequence[TrainState]
    graph_to_state: Sequence[Callable]
    state_to_graph: Sequence[Callable]
    alg_vars_from_graph: Sequence[Callable]
    system_configs: Sequence[Dict]
    composite_system_config: Dict
    Alambda: Array

    def setup(self):
        ########################################
        # TODO: test
        self.num_subsystems = len(self.system_configs)
        self.Es = [cfg['E'] for cfg in self.system_configs]
        self.Js = [cfg['J'] for cfg in self.system_configs]
        self.rs = [cfg['r'] for cfg in self.system_configs]
        self.B_bars = [cfg['B'] for cfg in self.system_configs]

        self.num_nodes = [cfg['num_nodes'] for cfg in self.system_configs]
        self.num_capacitors = [cfg['num_capacitors'] for cfg in self.system_configs]
        self.num_resistors = [cfg['num_resistors'] for cfg in self.system_configs]
        self.num_inductors = [cfg['num_inductors'] for cfg in self.system_configs]
        self.num_volt_sources = [cfg['num_volt_sources'] for cfg in self.system_configs]
        self.num_cur_sources = [cfg['num_cur_sources'] for cfg in self.system_configs]
        self.state_dims = [cfg['state_dim'] for cfg in self.system_configs]

        self.diff_indices = [cfg['diff_indices'] for cfg in self.system_configs]
        self.alg_indices = [cfg['alg_indices'] for cfg in self.system_configs]
        self.alg_eq_indices = [cfg['alg_eq_indices'] for cfg in self.system_configs]
        self.num_diff_vars = [len(cfg['diff_indices']) for cfg in self.system_configs]
        self.num_alg_vars = [len(cfg['alg_indices']) for cfg in self.system_configs]

        # Perform LU decomposition on Es[i] and invert decomposition matrices.
        self.P_invs = [
            jax.scipy.linalg.inv(jax.scipy.linalg.lu(E)[0])
            for E in self.Es
            ]
        self.L_invs = [
            jax.scipy.linalg.inv(jax.scipy.linalg.lu(E)[1])
            for E in self.Es
            ]
        self.U_invs = [
            jax.scipy.linalg.inv(
                jax.scipy.linalg.lu(E)[2][diff_ind][:,diff_ind]
                )
            for E, diff_ind in zip(self.Es, self.diff_indices)
            ]

        # A_{\lambda_i} \in \{-1, 0, 1\}^{n_{u_i} \times n_{\lambda}}
        # n_{u_i} is the dimension of the control for subsystem i, 
        # and n_{\lambda} is the number of coupling edges
        self.Alambdas = [
            self.Alambda[sum(self.num_nodes[:i]) : sum(self.num_nodes[:i+1])]
            for i in range(self.num_subsystems)
            ]
        
        self.num_lamb = len(self.Alambda.T)
        self.B_hats = get_B_hats(self.system_configs, self.Alambda)
        self.system_k_idx = get_system_k_idx(self.system_configs)
        
        ncc = sum(self.num_capacitors)
        nrc = sum(self.num_resistors)
        nlc = sum(self.num_inductors)
        nvc = sum(self.num_volt_sources)
        nic = sum(self.num_cur_sources)
        nec = sum(self.num_nodes)

        self.ncc = ncc
        self.nrc = nrc
        self.nlc = nlc
        self.nvc = nvc
        self.nic = nic
        self.nec = nec
        self.state_dim_c = ncc+nlc+nec+nvc+self.num_lamb

        # Equation (15) - there is typo on equation (18) 
        self.E = self.composite_system_config['E']
        self.J = self.composite_system_config['J']
        self.r = self.composite_system_config['r']
        self.B_bar = self.composite_system_config['B']
        self.diff_indices_c = self.composite_system_config['diff_indices']
        self.alg_indices_c = self.composite_system_config['alg_indices']
        self.alg_eq_indices_c = self.composite_system_config['alg_eq_indices']
        self.num_diff_vars_c = self.composite_system_config['num_diff_vars']
        self.num_alg_vars_c = self.composite_system_config['num_alg_vars']
        self.P_inv = self.composite_system_config['P_inv']
        self.L_inv = self.composite_system_config['L_inv']
        self.U_inv = self.composite_system_config['U_inv']

        self.ode_integrator = integrator_factory(self.ode_integration_method)

    def __call__(self, graphs, control, lamb, t, rng):
        '''
            1. Approximate lambda(0) using z(0), full DAE equations, and fsolve
                - TODO: right now assuming lambda(0) = 0
            2. Solve the (k-1) subsystems with coupling input u_hat_i = lambda from previous iteration
            3. Solve subsystem (k) with coupling input u_hat_k = sum_{i=1}^{k-1} A_lambda_i.T e_i 
               from previous or current iteration
            4. Repeat from 2.
        '''
        controls = explicit_unbatch_control(control, self.system_configs)
        states = [g_to_s(g) for g_to_s, g in zip(self.graph_to_state, graphs)]
        nc = self.ncc
        nl = self.nlc
        ne = self.nec
        nv = self.nvc
        qs = []
        phis = []
        es = []
        jvs = []


        # TODO: state_to_(qs, phis, es, jvs): Sequence[Callable] -> Sequence[Sequence]
        for i, state in enumerate(states):
            qs.append(state[0 : self.num_capacitors[i]])
            phis.append(state[self.num_capacitors[i] : self.num_capacitors[i]+self.num_inductors[i]])
            es.append(state[
                self.num_capacitors[i]+self.num_inductors[i] :
                self.num_capacitors[i]+self.num_inductors[i]+self.num_nodes[i]
            ])
            jvs.append(state[
                self.num_capacitors[i]+self.num_inductors[i]+self.num_nodes[i] :
                self.num_capacitors[i]+self.num_inductors[i]+self.num_nodes[i]+self.num_volt_sources[i]
            ])
        
        q = jnp.concatenate(qs)
        phi = jnp.concatenate(phis)
        e = jnp.concatenate(es) 
        jv = jnp.concatenate(jvs)
        state = jnp.concatenate((q, phi, e, jv))

        # Jacobian-type approach
        def get_coupling_input(lamb):
            u_hats = []
            for i in range(self.num_subsystems):
                if i == self.system_k_idx:
                    coupling_constraint = jnp.sum(
                        jnp.array([jnp.matmul(Al_i.T, e_i) for Al_i, e_i in zip(self.Alambdas, es)])
                    )

                    u_hats.append(jnp.array([coupling_constraint]))
                else:
                    u_hats.append(-lamb)
            return u_hats

        def H_from_state(x):
            """
            Returns the total Hamiltonian of the composite system and finds the next state for all subsystems            
            """

            # Get initial u_hat
            lamb = x[-self.num_lamb:]
            u_hats = get_coupling_input(lamb)
            states = []
            # Extract the subsystem states from the composite system state
            for i in range(self.num_subsystems):
                q_i = x[sum(self.num_capacitors[:i]) : sum(self.num_capacitors[:i+1])]
                phi_i = x[
                    nc+sum(self.num_inductors[:i]) : 
                    nc+sum(self.num_inductors[:i+1])
                ]
                e_i = x[
                    nc+nl+sum(self.num_nodes[:i]) :
                    nc+nl+sum(self.num_nodes[:i+1]) : 
                ]
                jv_i = x[
                    nc+nl+ne+sum(self.num_volt_sources[:i]) : 
                    nc+nl+ne+sum(self.num_volt_sources[:i+1])
                ]

                state_i = jnp.concatenate((q_i, phi_i, e_i, jv_i))
                states.append(state_i)
            
            # Get Hamiltonian predictions for all subsystems.
            # For all of the (k-1) subsystems, also get the next subsystem state.
            next_graphs = []
            for i in range(self.num_subsystems):
                state_i = states[i]
                H_i, next_graph_i = H_from_state_i(state_i, i)

                if i != self.system_k_idx:
                    x_i = state_i[self.diff_indices[i]]
                    next_y_i = self.alg_vars_from_graph[i](next_graph_i, self.alg_indices[i])
                    next_x_i = self.ode_integrator(
                        partial(neural_dae_i, params=(i, next_y_i, u_hats)), x_i, t, self.dt, self.T
                    )
                    next_state_i = jnp.zeros((self.state_dims[i]))
                    next_state_i = next_state_i.at[self.diff_indices[i]].set(next_x_i)
                    next_state_i = next_state_i.at[self.alg_indices[i]].set(next_y_i)
                    next_graph_i = self.state_to_graph[i](
                        next_state_i, controls[i], globals=next_graph_i.globals
                        )
                
                next_graphs.append(next_graph_i)
            
            H = sum([next_graph.globals[0] for next_graph in next_graphs])
            return H, next_graphs
        
        def H_from_state_i(state_i, i):
            """
                Returns the Hamiltonian prediction H_i and graph_i for subsystem i
            """
            def get_H_k():
                # For the known subsystem k
                phi = state_i[0]
                H_k = 0.5 * phi**2
                graph_k = jraph.GraphsTuple(nodes=jnp.array([[0, 0, 0, 0]]),
                                            edges=jnp.array([[0, 0]]),
                                            globals=jnp.array([H_k]),
                                            receivers=jnp.array([0]),
                                            senders=jnp.array([0]),
                                            n_node=jnp.array([0]),
                                            n_edge=jnp.array([0]))
                return H_k, graph_k
            def get_H_i():
                graph_i = self.state_to_graph[i](state=state_i, control=controls[i])
                intermediate_graph_i = self.train_states[i].apply_fn(
                    self.train_states[i].params, graph_i, controls[i], t, rng
                )
                H_i = intermediate_graph_i.globals[0]
                return H_i, intermediate_graph_i
            
            return get_H_k() if i == self.system_k_idx else get_H_i()

        def dynamics_function_i(x, t, params):
            i, u_hats = params
            e_i = x[
                self.num_capacitors[i]+self.num_inductors[i] :
                self.num_capacitors[i]+self.num_inductors[i]+self.num_nodes[i]
            ]
            jv_i = x[
                self.num_capacitors[i]+self.num_inductors[i]+self.num_nodes[i] : 
                self.num_capacitors[i]+self.num_inductors[i]+self.num_nodes[i]+self.num_volt_sources[i]
            ]
            dH_i, _ = jax.grad(H_from_state_i, argnums=0, has_aux=True)(x, i)
            dHq = dH_i[0 : self.num_capacitors[i]]
            dHphi = dH_i[self.num_capacitors[i] : self.num_capacitors[i]+self.num_inductors[i]]

            if i == self.system_k_idx:
                lamb = x[-self.num_lamb:]
                z_i = jnp.concatenate((e_i, dHphi, dHq, jv_i, lamb))
            else:
                z_i = jnp.concatenate((e_i, dHphi, dHq, jv_i))

            return jnp.matmul(self.Js[i], z_i) - self.rs[i](z_i) + jnp.matmul(self.B_bars[i], controls[i]) + jnp.matmul(self.B_hats[i], u_hats[i])
        
        def neural_dae_i(x, t, params):
            i, next_y, u_hats = params
            z = jnp.concatenate((x, next_y))
            daes = dynamics_function_i(z, t, (i, u_hats))
            return self.U_invs[i] @ (self.L_invs[i] @ self.P_invs[i] @ daes)[self.diff_indices[i]]
        
        # Step 1: solve for lambda_0 by solving full system on the first iteration
        if lamb is None:
            def monolithic_dynamics_function(state, t):
                dH, _ = jax.grad(H_from_state, has_aux=True)(state)
                e = state[nc+nl : nc+nl+ne]
                jv = state[nc+nl+ne : nc+nl+ne+nv]
                dHq = dH[0 : nc]
                dHphi = dH[nc : nc+nl]
                lamb = state[nc+nl+ne+nv : ]
                z = jnp.concatenate((e, dHphi, dHq, jv, lamb))
                result = jnp.matmul(self.J, z) - self.r(z) + jnp.matmul(self.B_bar, control)
                return result

            @jax.jit
            def monolithic_f(x, y, t, params):
                state = jnp.concatenate((x,y))
                monolithic_daes = monolithic_dynamics_function(state, t)
                return monolithic_daes[self.diff_indices_c]
            
            @jax.jit
            def monolithic_g(x, y, t, params):
                state = jnp.concatenate((x,y))
                monolithic_daes = monolithic_dynamics_function(state, t)
                return monolithic_daes[self.alg_indices_c]
            
            def test_g(x, y, t, params):
                state = jnp.concatenate((x,y))
                monolithic_daes = monolithic_dynamics_function(state, t)
                return jnp.sum(monolithic_daes[self.alg_eq_indices_c] ** 2)

            # This is initial guess for lambda
            lamb0 = jnp.zeros((self.num_lamb)) 
            # t1 = time.time()
            # print(f"Initial dynamics function {monolithic_dynamics_function(state, t)}")
            # t2 = time.time()
            # print(f"The monolithic dynamics function took {t2 - t1}")
            # TODO: BTW, if this works every iteration, then we can just get next_state from here...

            # ynew = minimize(lambda y : test_g(z0[self.diff_indices_c], y, t, None), z0[self.alg_indices_c])
            # if ynew.success:
            #     print(f"New algebraic var: {ynew}")
            #     z0 = jnp.concatenate((z0[self.diff_indices_c], ynew))
            # else:
            #     print(f"Optimization failed")
                
            full_system_solver = DAESolver(
                    monolithic_f, monolithic_g, self.diff_indices_c, self.alg_indices_c
                )
            # # next_state = full_system_solver.solve_dae(z0, jnp.array([self.dt]), params=None, y0_tol=10)[0]
            next_state = full_system_solver.solve_dae_one_timestep_rk4(state, t, self.dt, params=None)
            lamb = next_state[-self.num_lamb:]

            jax.debug.print('lamb')

            lamb = lamb0
                
        def fi(x, y, t, params):
            i, u_hats = params
            state = jnp.concatenate((x, y))
            dae_i = dynamics_function_i(state, t, params) 
            if i == self.system_k_idx:
                return dae_i[self.diff_indices[i]]
            else:
                return self.U_invs[i] @ (self.L_invs[i] @ self.P_invs[i] @ dae_i)[self.diff_indices[i]]
        
        def gi(x, y, t, params):
            i, u_hats = params
            state = jnp.concatenate((x, y))
            residuals_i = dynamics_function_i(state, t, params)[self.alg_eq_indices[i]]
            return residuals_i
        

        H, next_graphs = H_from_state(state)
        next_states = []
        next_es = []
        for i in range(self.num_subsystems):
            if i == self.system_k_idx:
                pass
            else:
                next_state_i = self.graph_to_state[i](next_graphs[i])
                next_e_i = next_state_i[
                    self.num_capacitors[i]+self.num_inductors[i] : 
                    self.num_capacitors[i]+self.num_inductors[i]+self.num_nodes[i]
                ]
                next_states.append(next_state_i)
                next_es.append(next_e_i)

        # Gauss-Seidel approach
        # u_hats[self.system_k_idx] = jnp.sum(jnp.matmul(self.Alambdas.T, next_es))

        # Step 3: Solve for the extended next state of subsystem k (w/ lambda) using DAE solver

        # TODO: try to make PH system matrices for system k with incidence matrices w/o gnd node
        u_hats = get_coupling_input(lamb)

        k = self.system_k_idx
        dae_solver = DAESolver(
            fi, gi, self.diff_indices[k], self.alg_indices[k]
            )
        state_k_ext = jnp.concatenate((states[k], lamb))
        params=(k, u_hats)

        # TODO: write out equations to make sure it makes sense
        # yknew, infodict, ier, mesg = fsolve(lambda yy : dae_solver.g(x0, yy, t, params), y0, full_output=True)

        # if ier != 1:
        #     # throw an error if the algebraic states are not consistent.
        #     raise ValueError("Initial algebraic states were inconsistent. fsolve returned {}".format(mesg))
        
        # state_k_ext = state_k_ext.at[self.alg_indices[k]].set(yknew)

        # next_state_k_ext = dae_solver.solve_dae(state_k_ext, jnp.array([self.dt]), params=params)
        next_state_k_ext = dae_solver.solve_dae_one_timestep_rk4(state_k_ext, t, self.dt, params)

        next_lamb = next_state_k_ext[-self.num_lamb:]

        next_state_k = next_state_k_ext[:-self.num_lamb]

        next_states.insert(k, next_state_k)
        next_graphs[self.system_k_idx] = self.state_to_graph[k](
                next_state_k, controls[k], set_ground_and_control=True,
                globals=next_graphs[self.system_k_idx].globals
            )
        
        jax.debug.print('{}', next_graphs[self.system_k_idx])

        return next_graphs, next_lamb