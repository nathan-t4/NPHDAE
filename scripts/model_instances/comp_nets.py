import jraph
import jax
import flax.linen as nn
import jax.numpy as jnp
from typing import Callable
from flax.training.train_state import TrainState
from flax.typing import Array
from helpers.integrator_factory import integrator_factory
from utils.graph_utils import *
from utils.jax_utils import *
from utils.models_utils import *
from utils.gnn_utils import *
from utils.comp_utils import *
from dae_solver.index1_semi_explicit_flax import DAESolver

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
    
class CompLCGNS(nn.Module):
    integration_method: str
    dt: float
    T: int
    state_one: TrainState
    state_two: TrainState 
    graph_to_state_one: Callable
    graph_to_state_two: Callable
    state_to_graph_one: Callable
    state_to_graph_two: Callable
    system_one_config: Dict
    system_two_config: Dict
    Alambda: Array

    def setup(self):
        self.AC1 = self.system_one_config['AC']
        self.AL1 = self.system_one_config['AL']
        self.AR1 = self.system_one_config['AR']
        self.AV1 = self.system_one_config['AV']
        self.AI1 = self.system_one_config['AI']
        
        self.num_nodes_1 = self.system_one_config['num_nodes']
        self.num_capacitors_1 = self.system_one_config['num_capacitors']
        self.num_inductors_1 = self.system_one_config['num_inductors']
        self.num_resistors_1 = self.system_one_config['num_resistors']
        self.num_volt_sources_1 = self.system_one_config['num_volt_sources']
        self.num_cur_sources_1 = self.system_one_config['num_cur_sources']
        self.state_one_dim = self.system_one_config['state_dim']

        self.AC2 = self.system_two_config['AC']
        self.AL2 = self.system_two_config['AL']
        self.AR2 = self.system_two_config['AR']
        self.AV2 = self.system_two_config['AV']
        self.AI2 = self.system_two_config['AI']

        self.num_nodes_2 = self.system_two_config['num_nodes']
        self.num_capacitors_2 = self.system_two_config['num_capacitors']
        self.num_inductors_2 = self.system_two_config['num_inductors']
        self.num_resistors_2 = self.system_two_config['num_resistors']
        self.num_volt_sources_2 = self.system_two_config['num_volt_sources']
        self.num_cur_sources_2 = self.system_two_config['num_cur_sources']
        self.state_two_dim = self.system_two_config['state_dim']

        self.AC = jax.scipy.linalg.block_diag(self.AC1, self.AC2) # Generalize to n systems
        self.AL = jax.scipy.linalg.block_diag(self.AL1, self.AL2)
        self.AR = jax.scipy.linalg.block_diag(self.AR1, self.AR2)
        self.AV = jax.scipy.linalg.block_diag(self.AV1, self.AV2)
        self.AI = jax.scipy.linalg.block_diag(self.AI1, self.AI2)
        self.Alambda1 = self.Alambda[0:len(self.state_one_dim)]
        self.Alambda2 = self.Alambda[len(self.state_one_dim):len(self.state_one_dim)+len(self.state_two_dim)]

        self.num_capacitors = self.num_capacitors_1 + self.num_capacitors_2
        self.num_inductors = self.num_inductors_1 + self.num_inductors_2
        self.num_nodes = self.num_nodes_1 + self.num_nodes_2
        self.num_volt_sources = self.num_volt_sources_1 + self.num_volt_sources_2
        self.num_cur_sources = self.num_cur_sources_1 + self.num_cur_sources_2
        self.num_lamb = len(self.Alambda.T)
        self.num_lamb1 = len(self.Alambda1.T)
        self.num_lamb2 = len(self.Alambda2.T)

        self.E1 = self.system_one_config['E']
        self.J1 = self.system_one_config['J']
        self.r1 = self.system_one_config['r']
        self.B_bar1 = self.system_one_config['B']
        self.B_hat1 = jnp.concatenate(
            self.Alambda1, 
            jnp.zeros((self.num_inductors_1+self.num_nodes_1+self.num_volt_sources_1, self.num_lamb1))
        )
        P1, L1, U1 = jax.scipy.linalg.lu(self.E1)
        self.P1_inv = jax.scipy.linalg.inv(P1)
        self.L1_inv = jax.scipy.linalg.inv(L1)
        self.U1_inv = jax.scipy.linalg.inv(U1[self.diff_indices_1][:,self.diff_indices_1])

        self.E2 = self.system_two_config['E']
        self.J2 = self.system_two_config['J']
        self.r2 = self.system_two_config['r']
        self.B_bar2 = self.system_two_config['B']
        self.B_hat2 = jnp.stack(
            jnp.zeros((self.state_two_dim, self.num_lamb1)),
            jnp.eye(self.num_lamb1)
        )
        P2, L2, U2 = jax.scipy.linalg.lu(self.E2)
        self.P2_inv = jax.scipy.linalg.inv(P2)
        self.L2_inv = jax.scipy.linalg.inv(L2)
        self.U2_inv = jax.scipy.linalg.inv(U2[self.diff_indices_2][:,self.diff_indices_2])

        ########################################
        # Under development - generalize to n-systems
        num_systems = len(system_config)
        self.Es = [cfg['E'] for cfg in system_configs]
        self.Js = [cfg['J'] for cfg in system_configs]
        self.rs = [cfg['r'] for cfg in system_configs]
        self.B_bars = [cfg['B'] for cfg in system_configs]

        self.P_invs = []
        self.L_invs = []
        self.U_invs = []
        ########################################

        for i in range(num_systems):
            P, L, U = jax.scipy.linalg.lu(self.Es[i])
            self.P_invs.append(jax.scipy.linalg.inv(P))
            self.L_invs.append(jax.scipy.linalg.inv(L))
            self.U_invs.append(jax.scipy.linalg.inv(U[self.diff_indices[i]][:,self.diff_indices[i]]))


        composite_system_config = {
            'AC': jax.scipy.linalg.block_diag(self.AC1, self.AC2), # Generalize to n systems
            'AL': jax.scipy.linalg.block_diag(self.AL1, self.AL2),
            'AR': jax.scipy.linalg.block_diag(self.AR1, self.AR2),
            'AV': jax.scipy.linalg.block_diag(self.AV1, self.AV2),
            'AI': jax.scipy.linalg.block_diag(self.AI1, self.AI2),
            'Alambda1': self.Alambda[0:len(self.state_one_dim)],
            'Alambda2': self.Alambda[len(self.state_one_dim):len(self.state_one_dim)+len(self.state_two_dim)],
            'num_capacitors': self.num_capacitors_1 + self.num_capacitors_2,
            'num_inductors': self.num_inductors_1 + self.num_inductors_2,
            'num_nodes': self.num_nodes_1 + self.num_nodes_2,
            'num_volt_sources': self.num_volt_sources_1 + self.num_volt_sources_2,
            'num_cur_sources': self.num_cur_sources_1 + self.num_cur_sources_2,
            'num_lamb': len(self.Alambda.T),
            'num_lamb1': len(self.Alambda1.T),
            'num_lamb2': len(self.Alambda2.T),
        }

        # TODO
        self.J = get_J_matrix(composite_system_config)
        self.E = get_E_matrix(composite_system_config)
        self.r = jnp.concatenate((self.r1, self.r2))
        self.B_bar = get_B_bar_matrix(composite_system_config)
        P, L, U = jax.scipy.linalg.lu(self.E)
        self.P_inv = jax.scipy.linalg.inv(P)
        self.L_inv = jax.scipy.linalg.inv(L)
        self.U_inv = jax.scipy.linalg.inv(U[self.diff_indices][:,self.diff_indices])

        self.ode_integrator = integrator_factory(self.integration_method)

        self.diff_indices_1 = self.system_one_config['diff_indices']
        self.alg_indices_1 = self.system_one_config['alg_indices']
        self.num_diff_vars_1 = len(self.diff_indices_1)
        self.num_alg_vars_1 = len(self.alg_indices_1)

        self.diff_indices_2 = self.system_two_config['diff_indices']
        self.alg_indices_2 = self.system_two_config['alg_indices']
        self.num_diff_vars_2 = len(self.diff_indices_2)
        self.num_alg_vars_2 = len(self.alg_indices_2)

        self.diff_indices = jnp.concatenate((self.diff_indices_1, self.state_one_dim + self.diff_indices_2))
        self.alg_indices = jnp.concatenate((self.alg_indices_1, self.state_one_dim + self.alg_indices_2))
        self.num_diff_vars = self.num_diff_vars_1 + self.num_diff_vars_2
        self.num_alg_vars = self.num_alg_vars_1 + self.num_alg_vars_2

    def __call__(self, graph, control, t, rng):
        '''
            1. Approximate lambda(0) using z(0), full DAE equations, and fsolve
            2. Solve subsystem 1 with coupling input u_hat_1 = lambda from previous iteration
            3. Solve subsystem 2 with coupling input u_hat_2 = sum_{i=1}^{k-1} A_lambda_i e_i 
               from previous or current iteration
            4. Repeat from 2.
        '''
        graph1, graph2 = explicit_unbatch_graph(graph, self.system_one_config, self.system_two_config)

        state1 = self.graph_to_state_one(graph1)
        state2 = self.graph_to_state_two(graph2)
        
        q1 = state1[0:self.num_capacitors_1]
        q2 = state2[0:self.num_capacitors_2]
        phi1 = state1[self.num_capacitors_1:self.num_capacitors_1+self.num_inductors_1]
        phi2 = state2[self.num_capacitors_2:self.num_capacitors_2+self.num_inductors_2]
        e1 = state1[
            self.num_capacitors_1+self.num_inductors_1 :
            self.num_capacitors_1+self.num_inductors_1+self.num_nodes_1
        ]
        e2 = state2[
            self.num_capacitors_2+self.num_inductors_2 : 
            self.num_capacitors_2+self.num_inductors_2+self.num_nodes_2
        ]
        jv1 = state1[
            self.num_capacitors_1+self.num_inductors_1+self.num_nodes_1 : 
            len(state1)
        ]
        jv2 = state2[
            self.num_capacitors_2+self.num_inductors_2+self.num_nodes_2 : 
            len(state2)
        ]

        q = jnp.concatenate((q1, q2))
        phi = jnp.concatenate((phi1, phi2))
        e = jnp.concatenate((e1, e2))
        jv = jnp.concatenate((jv1, jv2))
        lamb = jnp.zeros(len(self.Alambda)) # Solve for current lambda

        state = jnp.stack((q, phi, e, jv, lamb)) # l initialized to zero, |l| = number of added edges

        control1 = control[0:self.num_cur_sources_1+self.num_volt_sources_1]
        control2 = control[
            self.num_cur_sources_1+self.num_volt_sources_1 : 
            self.num_cur_sources_1+self.num_volt_sources_1+self.num_cur_sources_2+self.num_volt_sources_2
        ]

        # Step 1: solve for lambda_0 by solving full system on first iteration
        if lamb is None:
            def monolithic_dynamics_function(state, t):
                dH, _ = jax.grad(H_from_state)(state)
                e = state[
                    self.num_capacitors+self.num_inductors :
                    self.num_capacitors+self.num_inductors+self.num_nodes
                ]
                jv = state[
                    self.num_capacitors+self.num_inductors+self.num_nodes :
                    self.num_capacitors+self.num_inductors+self.num_nodes+self.num_volt_sources
                ]
                dHq = dH[:self.num_capacitors]
                dHphi = dH[self.num_capacitors : self.num_capacitors+self.num_inductors]
                z = jnp.stack((e, dHphi, dHq, jv, lamb))
                return jnp.matmul(self.J, z) - self.r(z) + jnp.matmul(self.B_bar, control)

            def monolithic_f(x, y, t, params):
                monolithic_daes = monolithic_dynamics_function(jnp.concatenate((x, y)), t)
                return self.U_inv @ (self.L_inv @ self.P_inv @ monolithic_daes)[self.diff_indices]
            
            def monolithic_g(x, y, t, params):
                monolithic_daes = monolithic_dynamics_function(jnp.concatenate((x, y)), t)
                return monolithic_daes[self.alg_indices]
            
            # TODO: BTW, if this works every iteration, then we can just get next_state from here...
            full_system_solver = DAESolver(monolithic_f, monolithic_g, self.num_diff_vars, self.num_alg_vars)
            next_state = full_system_solver.solve_dae_one_timestep_rk4(state, t, self.dt, params=None)
            lamb = next_state[-self.num_lamb]
        
        # Jacobian-type approach
        u_hat1 = -lamb
        u_hat2 = self.Alambda1.T @ e1

        def H_from_state(x):
            x1 = x[0:self.state_one_dim]
            x2 = x[self.state_one_dim:self.state_one_dim+self.state_two_dim]

            # Step 2: Predict the next state of subsystem 1 using GNN1
            graph1 = self.graph_from_state_one(state=x1)
            intermediate_graph1 = self.state_one.apply_fn(self.state_one.params, graph1, control1, rng) 
            # TODO: add extra input to GNN for coupling input

            next_y1 = self.get_algebraic_vars_1(intermediate_graph1)
            next_x1 = self.ode_integrator(partial(neural_dae_1, next_y1=next_y1), x1, t, self.dt, self.T)
            next_state1 = jnp.concatenate((next_x1, next_y1))
            next_graph1 = self.state_to_graph_one(next_state1)

            # We will just use the Hamiltonian as predicted by GNN2, not the next state
            graph2 = self.graph_from_state_two(state=x2)
            next_graph2 = self.state_two.apply_fn(self.state_two.params, graph2, control2, rng)

            H1 = next_graph1.globals[0]
            H2 = next_graph2.globals[0]

            H = H1 + H2

            return H, (next_graph1, next_graph2)
        
        H, (next_graph1, next_graph2) = H_from_state(state)

        # Gauss-Seidel-type approach
        # next_state1 = self.graph_to_state_one(next_graph1)
        # next_e1 = next_state1[
        #     self.num_capacitors_1+self.num_inductors_1 :
        #     self.num_capacitors_1+self.num_inductors_1+self.num_nodes_1
        # ]
        # u_hat2 = self.Alambda1.T @ next_e1
        
        def dynamics_function_1(x, t, aux_data):
            x1 = x[0:self.state_one_dim]
            dH, _ = jax.grad(H_from_state, has_aux=True)(x)
            dH1 = dH[:len(x1)]
            dH1q = dH1[:self.num_capacitors_1]
            dH1phi = dH1[self.num_capacitors_1 : self.num_capacitors_1 + self.num_inductors_1]
            z1 = jnp.stack(e1, dH1phi, dH1q, jv1)

            return jnp.matmul(self.J1, z1) - self.r1(z1) + jnp.matmul(self.B_bar1, control1) + jnp.matmul(self.B_hat1, u_hat1)
        
        def neural_dae_1(x1, next_y1, t):
            z = jnp.concatenate((x1, next_y1))
            daes = dynamics_function_1(z, t, None)
            return self.U1_inv @ (self.L1_inv @ self.P1_inv @ daes)[self.diff_indices1]
        
        # def f1(x1, y1, t, aux_data):
        #     daes = dynamics_function_1(jnp.concatenate((x1, y1)), t, aux_data)
        #     return self.U1_inv @ (self.L1_inv @ self.P1_inv @ daes)[self.diff_indices1]

        # def g1(x1, y1, t, aux_data):
        #     residuals1 = dynamics_function_1(jnp.concatenate((x1, y1)), t, aux_data)[self.alg_indices1]
        #     return residuals1
        
        def dynamics_function_2(x, t, aux_data):
            dH, _ = jax.grad(H_from_state, has_aux=True)(x)  
            dH2 = dH[self.state_one_dim:self.state_one_dim+self.state_two_dim]
            z2 = jnp.stack(e2, dH2[self.num_capacitors_2:self.num_capacitors_2+self.num_inductors_2], dH2[0:self.num_capacitors_2], jv2, lamb)

            return jnp.matmul(self.J2, z2).squeeze() - self.r2(z2) + jnp.matmul(self.B_bar2, control2) + jnp.matmul(self.B_hat2, u_hat2)
        
        def f2(x2, y2, t, aux_data):
            dae2 = dynamics_function_2(jnp.concatenate((x2, y2)), t, aux_data)
            return self.U2_inv @ (self.L2_inv @ self.P2_inv @ dae2)[self.diff_indices2]
        
        def g2(x2, y2, t, aux_data):
            residuals2 = dynamics_function_2(jnp.concatenate((x2, y2)), t, aux_data)[self.alg_indices_2]
            return residuals2
        

        # Step 3: Solve for the extended next state of subsystem 2 (w/ lambda) using DAE solver
        next_state2 = None
        if self.integration_method == 'dae':
            dae_solver = DAESolver(f2, g2, self.num_diff_vars_2, self.num_alg_vars_2)
            state2_ext = jnp.concatenate((state2, lamb))
            next_state2_ext = dae_solver.solve_dae_one_timestep_rk4(state2_ext, t, self.dt, params=None) 
            next_lamb = next_state2[-self.num_lamb]

        next_state1 = self.graph_to_state_one(next_graph1)
        next_e1 = next_state1[
            self.num_capacitors_1+self.num_inductors_1 : 
            self.num_capacitors_1+self.num_inductors_1+self.num_nodes_1
        ]

        next_state2 = next_state2_ext[:-self.num_lamb]
        
        next_graph2 = self.graph_from_state(state=next_state2, 
                                            system_params=False, 
                                            set_nodes=True,
                                            set_ground_and_control=True, 
                                            globals=next_graph2.globals)
        
        next_graph = jraph.batch((next_graph1, next_graph2))
        next_u_hat1 = - next_lamb
        next_u_hat2 = self.Alambda1.T @ next_e1
        
        return next_graph, next_u_hat1, next_u_hat2