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
from scripts.model_instances.ph_gns import PHGNS_NDAE


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

        self.num_nodes = jnp.array([cfg['num_nodes'] for cfg in self.system_configs])
        self.num_capacitors = jnp.array([cfg['num_capacitors'] for cfg in self.system_configs])
        self.num_resistors = jnp.array([cfg['num_resistors'] for cfg in self.system_configs])
        self.num_inductors = jnp.array([cfg['num_inductors'] for cfg in self.system_configs])
        self.num_volt_sources = jnp.array([cfg['num_volt_sources'] for cfg in self.system_configs])
        self.num_cur_sources = jnp.array([cfg['num_cur_sources'] for cfg in self.system_configs])
        self.state_dims = jnp.array([cfg['state_dim'] for cfg in self.system_configs])

        self.diff_indices = [cfg['diff_indices'] for cfg in self.system_configs]
        self.alg_indices = [cfg['alg_indices'] for cfg in self.system_configs]
        self.alg_eq_indices = [cfg['alg_eq_indices'] for cfg in self.system_configs]
        self.num_diff_vars = [cfg['num_diff_vars'] for cfg in self.system_configs]
        self.num_alg_vars = [cfg['num_alg_vars'] for cfg in self.system_configs]

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
        self.Alambdas = self.composite_system_config['Alambdas']

        
        self.num_lamb = len(self.Alambda.T)
        self.B_hats = get_B_hats(self.system_configs, self.Alambda)
        self.system_k_idx = np.array(get_system_k_idx(self.system_configs))

        self.ncc = sum(self.num_capacitors)
        self.nrc = sum(self.num_resistors)
        self.nlc = sum(self.num_inductors)
        self.nvc = sum(self.num_volt_sources)
        self.nic = sum(self.num_cur_sources)
        self.nec = sum(self.num_nodes) - self.num_lamb
        self.state_dim_c = self.ncc+self.nlc+self.nec+self.nvc+self.num_lamb

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

        # CHECK THAT Js[1] is valid... If it is and it doesn't work then there is something wrong with 
        jax.debug.print('J1 {}', self.Js[1])
        jax.debug.print('E1 {}', self.Es[1])
        jax.debug.print('B1 {}', self.B_bars[1])
        jax.debug.print('B_hat 1 {}', self.B_hats[1])


        self.subsystem_to_composite_state = self.composite_system_config['subsystem_to_composite_state']
        self.composite_to_subsystem_states = self.composite_system_config['composite_to_subsystem_states']
        self.get_coupling_input = self.composite_system_config['get_coupling_input']
        self.get_coupling_input = partial(self.get_coupling_input, system_k_idx=self.system_k_idx)

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

        state, (qs, phis, es, jvs) = self.subsystem_to_composite_state(states, self.Alambda)
        
        def H_from_state(x):
            """
            Returns the total Hamiltonian of the composite system and finds the next state for all subsystems 

            H_from_state: 
                full_composite_state (including lambdas) -> composite_Hamiltonian, next_graphs           
            """
            # Get initial u_hat
            lamb = x[-self.num_lamb:]
            # Extract the subsystem states from the composite system state
            states, (qs, phis, es, jvs) = self.composite_to_subsystem_states(x)
            u_hats = self.get_coupling_input(lamb, es)
            # Get Hamiltonian predictions for all subsystems.
            # For all of the (k-1) subsystems, also get the next subsystem state.
            # next_graphs = []

            # for state_i, diff_indices_i, alg_indices_i, s_to_g in zip(
            #     states, self.diff_indices, self.alg_indices, self.state_to_graph):
            #     # TODO: scan this, output y is (Hs, states), then sum(Hs) to get H
            #     if i != self.system_k_idx:
            #         graph_i = self.state_to_graph[i](state=state_i, control=controls[i])
            #         intermediate_graph_i = self.train_states[i].apply_fn(
            #             self.train_states[i].params, graph_i, controls[i], t, rng
            #         )
            #         x_i = state_i[diff_indices_i]
            #         next_y_i = self.alg_vars_from_graph[i](intermediate_graph_i, alg_indices_i)
            #         next_x_i = self.ode_integrator(
            #             partial(neural_dae_i, params=(i, next_y_i, u_hats)), x_i, t, self.dt, self.T
            #         )
            #         next_state_i = jnp.concatenate((next_x_i, next_y_i))
            #         next_graph_i = s_to_g(
            #             next_state_i, controls[i], globals=next_graph_i.globals
            #             )  
            #     else:
            #         # Hamiltonian of the k-th subsystem is known
            #         phi = state_i[0]
            #         H_i = 0.5 * phi**2
            #         next_graph_i = jraph.GraphsTuple(nodes=jnp.array([[0, 0, 0, 0]]),
            #                                     edges=jnp.array([[0, 0]]),
            #                                     globals=jnp.array([H_i]),
            #                                     receivers=jnp.array([0]),
            #                                     senders=jnp.array([0]),
            #                                     n_node=jnp.array([0]),
            #                                     n_edge=jnp.array([0]))
                
            #     next_graphs.append(next_graph_i)
            
            # H = sum([next_graph.globals[0] for next_graph in next_graphs])

            # For now, write this out explicitly....
            def get_H_0(state_0):
                graph_0 = self.state_to_graph[0](state=state_0, control=controls[0])
                intermediate_graph_0 = self.train_states[0].apply_fn(
                    self.train_states[0].params, graph_0, controls[0], t, rng
                )
                H_0 = intermediate_graph_0.globals[0]
                return H_0, intermediate_graph_0

            H_0, intermediate_graph_0 = get_H_0(states[0])
            x_0 = states[0][self.diff_indices[0]]
            next_y_0 = self.alg_vars_from_graph[0](intermediate_graph_0, self.alg_indices[0])

            def dynamics_function_0(state_0, t):
                e_0 = state_0[2 : 5]
                jv_0 = state_0[5 : 6]
                dH_0, _ = jax.grad(get_H_0, argnums=0, has_aux=True)(x)
                dHq = dH_0[0 : 1]
                dHphi = dH_0[1 : 2]

                z_0 = jnp.concatenate((e_0, dHphi, dHq, jv_0))

                return jnp.matmul(self.Js[0], z_0) - self.rs[0](z_0) + jnp.matmul(self.B_bars[0], controls[0]) + jnp.matmul(self.B_hats[0], u_hats[0])

            def neural_dae_0(x0, t):
                z0 = jnp.concatenate((x0, next_y_0))
                dae0 = dynamics_function_0(z0, t)
                return self.U_invs[0] @ (self.L_invs[0] @ self.P_invs[0] @ dae0)[self.diff_indices[0]]
            
            next_x_0 = self.ode_integrator(
                neural_dae_0, x_0, t, self.dt, self.T
            )

            # def f0(x, y, t, params):
            #     dae0 = dynamics_function_0(jnp.concatenate((x,y)), t, params)
            #     return self.U_invs[0] @ (self.L_invs[0] @ self.P_invs[0] @ dae0)[self.diff_indices[0]]
            
            # def g0(x, y, t, params):
            #     dae0 = dynamics_function_0(jnp.concatenate((x,y)), t, params)
            #     return dae0[self.alg_eq_indices[0]]
            
            # dae_solver_0 = DAESolver(f0, g0, self.diff_indices[0], self.alg_indices[0])
            # next_state_0 = dae_solver_0.solve_dae(states[0], jnp.array([t + self.dt]), params=None)[0]

            next_state_0 = jnp.concatenate((next_x_0, next_y_0))
            next_graph_0 = self.state_to_graph[0](
                next_state_0, controls[0], globals=intermediate_graph_0.globals
                )  

            phi = states[1][0]
            H_1 = 0.5 * phi**2
            next_graph_1 = jraph.GraphsTuple(nodes=jnp.zeros(graphs[1].nodes.shape),
                                            edges=jnp.zeros(graphs[1].edges.shape),
                                            globals=jnp.array([H_1]),
                                            receivers=graphs[1].receivers,
                                            senders=graphs[1].senders,
                                            n_node=graphs[1].n_node,
                                            n_edge=graphs[1].n_edge) 
            
            def get_H_2(state_2):
                graph_2 = self.state_to_graph[2](state=state_2, control=controls[2])
                intermediate_graph_2 = self.train_states[2].apply_fn(
                    self.train_states[2].params, graph_2, controls[2], t, rng
                )
                H_2 = intermediate_graph_2.globals[0]
                return H_2, intermediate_graph_2
            
            H_2, intermediate_graph_2 = get_H_2(states[2])
            x_2 = states[2][self.diff_indices[2]]
            next_y_2 = self.alg_vars_from_graph[2](intermediate_graph_2, self.alg_indices[2])

            def dynamics_function_2(state_2, t):
                e_2 = state_2[2 : 5]
                jv_2 = state_2[5 : 6]
                dH_2, _ = jax.grad(get_H_2, argnums=0, has_aux=True)(x)
                dHq = dH_2[0 : 1]
                dHphi = dH_2[1 : 2]

                z_2 = jnp.concatenate((e_2, dHphi, dHq, jv_2))

                return jnp.matmul(self.Js[2], z_2) - self.rs[2](z_2) + jnp.matmul(self.B_bars[2], controls[2]) + jnp.matmul(self.B_hats[2], u_hats[2])
            
            def neural_dae_2(x2, t):
                z2 = jnp.concatenate((x2, next_y_2))
                dae2 = dynamics_function_2(z2, t)
                return self.U_invs[2] @ (self.L_invs[2] @ self.P_invs[2] @ dae2)[self.diff_indices[2]]
            
            next_x_2 = self.ode_integrator(
                neural_dae_2, x_2, t, self.dt, self.T
            )

            next_state_2 = jnp.concatenate((next_x_2, next_y_2))

            # def f2(x, y, t, params):
            #     dae2 = dynamics_function_2(jnp.concatenate((x,y)), t, params)
            #     return self.U_invs[2] @ (self.L_invs[2] @ self.P_invs[2] @ dae2)[self.diff_indices[2]]
            
            # def g2(x, y, t, params):
            #     dae2 = dynamics_function_0(jnp.concatenate((x,y)), t, params)
            #     return dae2[self.alg_eq_indices[2]]
            
            # dae_solver_2 = DAESolver(f2, g2, self.diff_indices[2], self.alg_indices[2])
            # next_state_2 = dae_solver_2.solve_dae(states[2], jnp.array([t + self.dt]), params=None)[0]

            next_graph_2 = self.state_to_graph[2](
                next_state_2, controls[2], globals=intermediate_graph_2.globals
                )  
            
            H = jnp.sum(jnp.array([H_0, H_1, H_2]))
            next_graphs = [next_graph_0, next_graph_1, next_graph_2]
            return H, next_graphs
        
        def H_from_state_i(state_i, i):
            """
                Returns the Hamiltonian prediction H_i and graph_i for subsystem i
            """
            def get_H_k():
                # For the known subsystem k
                phi = state_i[0]
                H_k = 0.5 * phi**2
                graph_k = jraph.GraphsTuple(nodes=jnp.zeros(graphs[1].nodes.shape),
                                            edges=jnp.zeros(graphs[1].edges.shape),
                                            globals=jnp.array([H_k]),
                                            receivers=graphs[1].receivers,
                                            senders=graphs[1].senders,
                                            n_node=graphs[1].n_node,
                                            n_edge=graphs[1].n_edge)
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

            full_system_solver = DAESolver(
                    monolithic_f, test_g, self.diff_indices_c, self.alg_indices_c # or monolithic_g
                )
            z0 = jnp.concatenate((state, lamb0))
            next_state_ext = full_system_solver.solve_dae(z0, jnp.array([t + self.dt]), params=None, y0_tol=1e-4)[0]
            # next_state = full_system_solver.solve_dae_one_timestep_rk4(z0, t, self.dt, params=None)
            lamb = next_state_ext[-self.num_lamb:]
            next_state = next_state_ext[0 : -self.num_lamb]
            next_states = self.composite_to_subsystem_states(next_state)
            # jax.debug.print('test lamb {}', lamb)
            # lamb = lamb0
                
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
        

        monolithic_state = jnp.concatenate((state, lamb))
        H, next_graphs = H_from_state(monolithic_state)
        next_states = []
        next_es = []
        # for i in range(self.num_subsystems):
        #     if i == self.system_k_idx:
        #         pass
        #     else:
        #         next_state_i = self.graph_to_state[i](next_graphs[i])
        #         next_e_i = next_state_i[
        #             self.num_capacitors[i]+self.num_inductors[i] : 
        #             self.num_capacitors[i]+self.num_inductors[i]+self.num_nodes[i]
        #         ]
        #         next_states.append(next_state_i)
        #         next_es.append(next_e_i)


        next_states.append(self.graph_to_state[0](next_graphs[0]))
        next_states.append(None) # still need to find state[1]
        next_states.append(self.graph_to_state[2](next_graphs[2]))

        next_es.append(next_states[0][2:5])
        next_es.append(None)
        next_es.append(next_states[2][2:5])

        # Gauss-Seidel approach
        # u_hats[self.system_k_idx] = jnp.sum(jnp.matmul(self.Alambdas.T, next_es))

        def H_from_state_1(state_1):
            # For the known subsystem k
            phi = state_1[0]
            H_1 = 0.5 * phi**2
            graph_1 = jraph.GraphsTuple(nodes=jnp.zeros(graphs[1].nodes.shape),
                                        edges=jnp.zeros(graphs[1].edges.shape),
                                        globals=jnp.array([H_1]),
                                        receivers=graphs[1].receivers,
                                        senders=graphs[1].senders,
                                        n_node=graphs[1].n_node,
                                        n_edge=graphs[1].n_edge) 
            return H_1, graph_1

        def dynamics_function_1(x, t, params):
            # TODO: Something about the gradients is incorrect, causing Phi to not change...
            # TODO: maybe u_hats??? Assume it is given instead (YES!)
            # next_states = params
            # e_0 = next_states[0][2 : 5]
            # e_2 = next_states[2][2 : 5]
            e_1 = x[1 : 4]
            # es = [e_0, e_1, e_2]
            u_hats = params
            jv_1 = x[4 : 4]
            dH_1, _ = jax.grad(H_from_state_1, has_aux=True)(x)
            dHq = dH_1[0 : 0]
            dHphi = dH_1[0 : 1]
            lamb = x[4 : 6]
            # u_hats = self.get_coupling_input(lamb, es)
            z_1 = jnp.concatenate((e_1, dHphi, dHq, jv_1, lamb))

            return jnp.matmul(self.Js[1], z_1) - self.rs[1](z_1) + jnp.matmul(self.B_bars[1], controls[1]) + jnp.matmul(self.B_hats[1], u_hats[1])
        
        def f1(x, y, t, params):
            state = jnp.concatenate((x, y))
            dae_1 = dynamics_function_1(state, t, params) 
            test = self.U_invs[1] @ (self.L_invs[1] @ self.P_invs[1] @ dae_1)[self.diff_indices[1]]
            return dae_1[jnp.array([3])]

        def g1(x, y, t, params):
            state = jnp.concatenate((x, y))
            # residuals_1 = dynamics_function_1(state, t, params)[self.alg_eq_indices[1]]
            residuals_1 = dynamics_function_1(state, t, params)[jnp.array([0,1,2,4,5])]
            return residuals_1
        
        
        # Step 3: Solve for the extended next state of subsystem k (w/ lambda) using DAE solver
        k = self.system_k_idx
        dae_solver = DAESolver(
            f1, g1, self.diff_indices[k], self.alg_indices[k]
            )
        state_k_ext = jnp.concatenate((states[k], lamb))
        u_hats = self.get_coupling_input(lamb, es) # Jacobian
        # TODO: params was next_states (changed to u_hats)
        # next_state_k_ext = dae_solver.solve_dae(state_k_ext, jnp.array([t + self.dt]), params=u_hats)[0]
        next_state_k_ext = dae_solver.solve_dae(state_k_ext, jnp.array([t + self.dt]), params=u_hats)

        next_lamb = next_state_k_ext[-self.num_lamb : len(next_state_k_ext)]
        next_state_k = next_state_k_ext[0 : -self.num_lamb]
        residual_1 = jnp.sum(
            g1(next_state_k[self.diff_indices[1]], next_state_k[self.alg_indices[1]], t, params=u_hats) ** 2
        )

        next_states[k] = next_state_k
        H_1 = H_from_state_1(next_states[k])[0]
        globals_1 = jnp.array([H_1, residual_1])
        next_graphs[self.system_k_idx] = self.state_to_graph[k](next_state_k, controls[k], set_ground_and_control=True, globals=globals_1)
                
        return next_graphs, next_lamb
    

class MonolithicCompPHGNS(nn.Module):
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
        self.num_subsystems = len(self.system_configs)
        self.Es = [cfg['E'] for cfg in self.system_configs]
        self.Js = [cfg['J'] for cfg in self.system_configs]
        self.rs = [cfg['r'] for cfg in self.system_configs]
        self.B_bars = [cfg['B'] for cfg in self.system_configs]

        self.num_nodes = jnp.array([cfg['num_nodes'] for cfg in self.system_configs])
        self.num_capacitors = jnp.array([cfg['num_capacitors'] for cfg in self.system_configs])
        self.num_resistors = jnp.array([cfg['num_resistors'] for cfg in self.system_configs])
        self.num_inductors = jnp.array([cfg['num_inductors'] for cfg in self.system_configs])
        self.num_volt_sources = jnp.array([cfg['num_volt_sources'] for cfg in self.system_configs])
        self.num_cur_sources = jnp.array([cfg['num_cur_sources'] for cfg in self.system_configs])
        self.state_dims = jnp.array([cfg['state_dim'] for cfg in self.system_configs])

        self.diff_indices = [cfg['diff_indices'] for cfg in self.system_configs]
        self.alg_indices = [cfg['alg_indices'] for cfg in self.system_configs]
        self.alg_eq_indices = [cfg['alg_eq_indices'] for cfg in self.system_configs]
        self.num_diff_vars = [cfg['num_diff_vars'] for cfg in self.system_configs]
        self.num_alg_vars = [cfg['num_alg_vars'] for cfg in self.system_configs]

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
        self.Alambdas = self.composite_system_config['Alambdas']

        
        self.num_lamb = len(self.Alambda.T)
        self.B_hats = get_B_hats(self.system_configs, self.Alambda)
        self.system_k_idx = np.array(get_system_k_idx(self.system_configs))

        self.ncc = sum(self.num_capacitors)
        self.nrc = sum(self.num_resistors)
        self.nlc = sum(self.num_inductors)
        self.nvc = sum(self.num_volt_sources)
        self.nic = sum(self.num_cur_sources)
        self.nec = sum(self.num_nodes) - self.num_lamb
        self.state_dim_c = self.ncc+self.nlc+self.nec+self.nvc+self.num_lamb

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

        self.subsystem_to_composite_state = self.composite_system_config['subsystem_to_composite_state']
        self.composite_to_subsystem_states = self.composite_system_config['composite_to_subsystem_states']
        self.get_coupling_input = self.composite_system_config['get_coupling_input']
        self.get_coupling_input = partial(self.get_coupling_input, system_k_idx=self.system_k_idx)

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

        state, (qs, phis, es, jvs) = self.subsystem_to_composite_state(states, self.Alambda)
        
        @jax.jit
        def H_from_state(x):
            """
            Returns the total Hamiltonian of the composite system and finds the next state for all subsystems 

            H_from_state: 
                full_composite_state (including lambdas) -> composite_Hamiltonian, next_graphs           
            """
            # Get initial u_hat
            lamb = x[-self.num_lamb:]
            # Extract the subsystem states from the composite system state
            states, (qs, phis, es, jvs) = self.composite_to_subsystem_states(x)
            u_hats = self.get_coupling_input(lamb, es)
            # Get Hamiltonian predictions for all subsystems.
            # For all of the (k-1) subsystems, also get the next subsystem state.
            def get_H_0(state_0):
                graph_0 = self.state_to_graph[0](state=state_0, control=controls[0])
                intermediate_graph_0 = (self.train_states[0].apply_fn)(
                    self.train_states[0].params, graph_0, controls[0], t, rng
                )
                H_0 = intermediate_graph_0.globals[0]
                return H_0, intermediate_graph_0

            H_0, intermediate_graph_0 = get_H_0(states[0])
            x_0 = states[0][self.diff_indices[0]]
            next_y_0 = self.alg_vars_from_graph[0](intermediate_graph_0, self.alg_indices[0])

            def dynamics_function_0(state_0, t):
                e_0 = state_0[2 : 5]
                jv_0 = state_0[5 : 6]
                dH_0, _ = jax.grad(get_H_0, argnums=0, has_aux=True)(x)
                dHq = dH_0[0 : 1]
                dHphi = dH_0[1 : 2]

                z_0 = jnp.concatenate((e_0, dHphi, dHq, jv_0))

                return jnp.matmul(self.Js[0], z_0) - self.rs[0](z_0) + jnp.matmul(self.B_bars[0], controls[0]) + jnp.matmul(self.B_hats[0], u_hats[0])

            def neural_dae_0(x0, t):
                z0 = jnp.concatenate((x0, next_y_0))
                dae0 = dynamics_function_0(z0, t)
                return self.U_invs[0] @ (self.L_invs[0] @ self.P_invs[0] @ dae0)[self.diff_indices[0]]
            
            next_x_0 = self.ode_integrator(
                neural_dae_0, x_0, t, self.dt, self.T
            )
            next_state_0 = jnp.concatenate((next_x_0, next_y_0))
            next_graph_0 = self.state_to_graph[0](
                next_state_0, controls[0], globals=intermediate_graph_0.globals
                )  

            phi = states[1][0]
            H_1 = 0.5 * phi**2
            next_graph_1 = jraph.GraphsTuple(nodes=jnp.zeros(graphs[1].nodes.shape),
                                            edges=jnp.zeros(graphs[1].edges.shape),
                                            globals=jnp.array([H_1]),
                                            receivers=graphs[1].receivers,
                                            senders=graphs[1].senders,
                                            n_node=graphs[1].n_node,
                                            n_edge=graphs[1].n_edge) 
            
            def get_H_2(state_2):
                graph_2 = self.state_to_graph[2](state=state_2, control=controls[2])
                intermediate_graph_2 = (self.train_states[2].apply_fn)(
                    self.train_states[2].params, graph_2, controls[2], t, rng
                )
                H_2 = intermediate_graph_2.globals[0]
                return H_2, intermediate_graph_2
            
            H_2, intermediate_graph_2 = get_H_2(states[2])
            x_2 = states[2][self.diff_indices[2]]
            next_y_2 = self.alg_vars_from_graph[2](intermediate_graph_2, self.alg_indices[2])

            def dynamics_function_2(state_2, t):
                e_2 = state_2[2 : 5]
                jv_2 = state_2[5 : 6]
                dH_2, _ = jax.grad(get_H_2, argnums=0, has_aux=True)(x)
                dHq = dH_2[0 : 1]
                dHphi = dH_2[1 : 2]

                z_2 = jnp.concatenate((e_2, dHphi, dHq, jv_2))

                return jnp.matmul(self.Js[2], z_2) - self.rs[2](z_2) + jnp.matmul(self.B_bars[2], controls[2]) + jnp.matmul(self.B_hats[2], u_hats[2])
            
            def neural_dae_2(x2, t):
                z2 = jnp.concatenate((x2, next_y_2))
                dae2 = dynamics_function_2(z2, t)
                return self.U_invs[2] @ (self.L_invs[2] @ self.P_invs[2] @ dae2)[self.diff_indices[2]]
            
            next_x_2 = self.ode_integrator(
                neural_dae_2, x_2, t, self.dt, self.T
            )

            next_state_2 = jnp.concatenate((next_x_2, next_y_2))
            next_graph_2 = self.state_to_graph[2](
                next_state_2, controls[2], globals=intermediate_graph_2.globals
                )  
            
            H = jnp.sum(jnp.array([H_0, H_1, H_2]))
            next_graphs = [next_graph_0, next_graph_1, next_graph_2]
            return H, next_graphs
        
        def monolithic_dynamics_function(state, t):
            dH, _ = jax.grad(H_from_state, has_aux=True)(state)
            e = state[5 : 12]
            jv = state[12 : 14]
            dHq = dH[0 : 2]
            dHphi = dH[2 : 5]
            lamb = state[14 : 16]
            z = jnp.concatenate((e, dHphi, dHq, jv, lamb))
            result = jnp.matmul(self.J, z) - self.r(z) + jnp.matmul(self.B_bar, control)
            return result

        @jax.jit
        def monolithic_f(x, y, t, params):
            state = jnp.concatenate((x,y))
            monolithic_daes = monolithic_dynamics_function(state, t)
            return self.U_inv @ (self.L_inv @ self.P_inv @ monolithic_daes)[self.diff_indices_c]
        
        @jax.jit
        def monolithic_g(x, y, t, params):
            state = jnp.concatenate((x,y))
            monolithic_daes = monolithic_dynamics_function(state, t)
            return monolithic_daes[self.alg_indices_c]

        lamb0 = jnp.zeros((self.num_lamb)) if lamb == None else lamb

        full_system_solver = DAESolver(
                monolithic_f, monolithic_g, self.diff_indices_c, self.alg_indices_c # or monolithic_g
            )
        z0 = jnp.concatenate((state, lamb0))
        next_state_ext = full_system_solver.solve_dae(z0, jnp.array([t + self.dt]), params=None)[0]
        # next_state = full_system_solver.solve_dae_one_timestep_rk4(z0, t, self.dt, params=None)
        next_lamb = next_state_ext[-self.num_lamb:]
        next_state = next_state_ext[0 : -self.num_lamb]
        next_states = self.composite_to_subsystem_states(next_state)[0]
        H, intermediate_graphs = H_from_state(z0)

        next_graphs = []
        next_graphs.append(self.state_to_graph[0](next_states[0], control=controls[0], globals=intermediate_graphs[0].globals))
        next_graphs.append(self.state_to_graph[1](next_states[1], control=controls[1], globals=intermediate_graphs[1].globals))
        next_graphs.append(self.state_to_graph[2](next_states[2], control=controls[2], globals=intermediate_graphs[2].globals))

        return next_graphs, next_lamb