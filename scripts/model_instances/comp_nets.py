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
from scipy.optimize import fsolve


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
    Alambda: Array

    def setup(self):
        ########################################
        # TODO: test
        self.num_subsystems = len(self.system_configs)
        
        self.ACs = [cfg['AC'] for cfg in self.system_configs]
        self.ARs = [cfg['AR'] for cfg in self.system_configs]
        self.ALs = [cfg['AL'] for cfg in self.system_configs]
        self.AVs = [cfg['AV'] for cfg in self.system_configs]
        self.AIs = [cfg['AI'] for cfg in self.system_configs]

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
            self.Alambda[sum(self.num_nodes[:i])-i : sum(self.num_nodes[:i+1])-(i+1)]
            for i in range(self.num_subsystems)
            ]
        
        # Alambdas with ground
        # self.full_Alambdas = [fill_in_incidence_matrix(a) for a in self.Alambdas]
        self.num_lamb = len(self.Alambda.T)

        self.system_k_idx = -1

        # TODO: Define B_hat in get_system_config in gnn_utils
        self.B_hats = []
        for i, cfg in enumerate(self.system_configs):            
            # Find the index of the k-th system
            if cfg['is_k']:
                if self.system_k_idx > 0:
                    raise ValueError(f"Last system has already been set to {self.system_k_idx}. Make sure there is only one subsystem with subsystem_config['last_system'] = True")
                else:
                    self.system_k_idx = i
                    B_hat_k = jnp.concatenate((
                        jnp.zeros((self.state_dims[i]-self.num_lamb, 1)), jnp.ones((self.num_lamb,1))
                    ))
                    self.B_hats = [*self.B_hats, B_hat_k]
            else:
                B_hat_i = jnp.concatenate((
                    self.Alambdas[i], 
                    jnp.zeros((self.num_inductors[i]+self.num_capacitors[i]+self.num_volt_sources[i], self.num_lamb))
                ))
                B_hat_i = fill_in_incidence_matrix(B_hat_i) # append gnd node
                self.B_hats = [*self.B_hats, B_hat_i]
            
        assert(self.system_k_idx != -1), \
            "k-th system index has not been set. \
            Make sure that one system config in self.system_configs has cfg['is_k'] = True!"
        
        ncc = sum(self.num_capacitors)
        nrc = sum(self.num_resistors)
        nlc = sum(self.num_inductors)
        nvc = sum(self.num_volt_sources)
        nic = sum(self.num_cur_sources)
        # TODO: avoid recounting gnd
        nec = sum(self.num_nodes) - self.num_subsystems + 1 

        self.ncc = ncc
        self.nrc = nrc
        self.nlc = nlc
        self.nvc = nvc
        self.nic = nic
        self.nec = nec
        self.state_dim_c = ncc+nlc+nec+nvc+self.num_lamb

        # keep track of nonzero columns
        # First append ground nodes
        AC_c = []
        AR_c = []
        AL_c = []
        AV_c = []
        AI_c = []

        # pick out columns that are all zeros
        for i, (AC, AR, AL, AV, AI) in enumerate(
            zip(self.ACs, self.ARs, self.ALs, self.AVs, self.AIs)):
            if (AC != 0.0).any():
                AC_c = [*AC_c, i]
            if (AR != 0.0).any():
                AR_c = [*AR_c, i]
            if (AL != 0.0).any():
                AL_c = [*AL_c, i]
            if (AV != 0.0).any():
                AV_c = [*AV_c, i]
            if (AI != 0.0).any():
                AI_c = [*AI_c, i]

        # exclude rows corresponding to ground nodes!
        comp_AC = jax.scipy.linalg.block_diag(*[AC[1:,:] for AC in self.ACs])[:,AC_c]
        comp_AR = jax.scipy.linalg.block_diag(*[AR[1:,:] for AR in self.ARs])[:,AR_c]
        comp_AL = jax.scipy.linalg.block_diag(*[AL[1:,:] for AL in self.ALs])[:,AL_c]
        comp_AV = jax.scipy.linalg.block_diag(*[AV[1:,:] for AV in self.AVs])[:,AV_c]
        comp_AI = jax.scipy.linalg.block_diag(*[AI[1:,:] for AI in self.AIs])[:,AI_c]

        # fill in composite incidence matrices s.t. all columns have one 1 and one (-1)
        comp_AC = fill_in_incidence_matrix(comp_AC)
        comp_AR = fill_in_incidence_matrix(comp_AR)
        comp_AL = fill_in_incidence_matrix(comp_AL)
        comp_AV = fill_in_incidence_matrix(comp_AV)
        comp_AI = fill_in_incidence_matrix(comp_AI)
        # do the same for Alambda
        Alambda = fill_in_incidence_matrix(self.Alambda)

        # Equation (15) - there is typo on equation (18) 
        # Alambda_with_gnd = jnp.concatenate((self.full_Alambdas))

        # Create composite J matrix
        self.J =  jnp.zeros((self.state_dim_c, self.state_dim_c))
        self.J = self.J.at[0:nec, nec:nec+nlc].set(-comp_AL)
        self.J = self.J.at[0:nec, nec+nlc+ncc : nec+nlc+ncc+nvc].set(-comp_AV)
        self.J = self.J.at[0:nec, nec+nlc+ncc+nvc : nec+nlc+ncc+nvc+self.num_lamb].set(-Alambda)
        
        self.J = self.J.at[nec : nec+nlc, 0:nec].set(comp_AL.T)
        self.J = self.J.at[nec+nlc+ncc : nec+nlc+ncc+nvc, 0:nec].set(comp_AV.T)
        self.J = self.J.at[nec+nlc+ncc+nvc : nec+nlc+ncc+nvc+self.num_lamb, 0:nec].set(Alambda.T)

        # Create composite E matrix
        self.E = jnp.zeros((self.state_dim_c, self.state_dim_c))
        self.E = self.E.at[0:nec, 0:ncc].set(comp_AC)
        self.E = self.E.at[nec:nec+nlc, ncc:ncc+nlc].set(jnp.eye(len(comp_AL.T)))

        # Create composite r vector 
        def get_composite_r(z):
            g = lambda e : (comp_AR.T @ e) / 1.0 
            # z = [e, jl, uc, jv, lamb]
            e = z[0 : nec]
            jl = z[nec: nec+nlc]
            uc = z[
                nec+nlc : 
                nec+nlc+ncc
                ]
            jv = z[nec+nlc+ncc : nec+nlc+ncc+nvc]
            lamb = z[-self.num_lamb:]

            curr_through_resistors = jnp.linalg.matmul(comp_AR, g(e))
            charge_constraint = jnp.matmul(comp_AC.T, e) - uc

            diss = jnp.zeros((self.state_dim_c,))
            diss = diss.at[0:nec].set(curr_through_resistors)
            diss = diss.at[(nec+nlc):(nec+nlc+ncc)].set(charge_constraint)

            return diss
        
        self.r = get_composite_r

        # Create composite B_bar
        self.B_bar = jnp.zeros((self.state_dim_c, nic+nvc))
        self.B_bar = self.B_bar.at[0:nec, 0:nic].set(-comp_AI)
        self.B_bar = self.B_bar.at[nec+nlc+ncc : nec+nlc+ncc+nvc, nic : nic+nvc].set(-jnp.eye(nvc))

        # Find the indices corresponding to the differential and algebraic variables
        self.diff_indices_c, self.alg_indices_c = get_diff_and_alg_indices(self.E)
        self.num_diff_vars_c = sum(self.num_diff_vars)
        self.num_alg_vars_c = sum(self.num_alg_vars) # avoid recounting gnd

        # LU decomposition on composite E + get inverses of decomposition matrices
        P, L, U = jax.scipy.linalg.lu(self.E)
        self.P_inv = jax.scipy.linalg.inv(P)
        self.L_inv = jax.scipy.linalg.inv(L)
        self.U_inv = jax.scipy.linalg.inv(U[self.diff_indices_c][:,self.diff_indices_c])

        self.ode_integrator = integrator_factory(self.ode_integration_method)

    def __call__(self, graph, control, lamb, t, rng):
        '''
            1. Approximate lambda(0) using z(0), full DAE equations, and fsolve
                - TODO: right now assuming lambda(0) = 0
            2. Solve the (k-1) subsystems with coupling input u_hat_i = lambda from previous iteration
            3. Solve subsystem (k) with coupling input u_hat_k = sum_{i=1}^{k-1} A_lambda_i.T e_i 
               from previous or current iteration
            4. Repeat from 2.
        '''
        graphs = jraph.unbatch(graph)
        # graphs = explicit_unbatch_graph(graph, self.Alambda, self.system_configs)
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
        for i, state in enumerate(states):
            qs.append(state[0 : self.num_capacitors[i]])
            phis.append(state[self.num_capacitors[i] : self.num_capacitors[i]+self.num_inductors[i]])
            # exclude ground node! (which is the first node)
            es.append(state[
                self.num_capacitors[i]+self.num_inductors[i]+1 :
                self.num_capacitors[i]+self.num_inductors[i]+self.num_nodes[i]
            ])
            jvs.append(state[
                self.num_capacitors[i]+self.num_inductors[i]+self.num_nodes[i] :
                self.num_capacitors[i]+self.num_inductors[i]+self.num_nodes[i]+self.num_volt_sources[i]
            ])
        es.insert(0, jnp.array([0])) # append common ground node
        q = jnp.concatenate(qs)
        phi = jnp.concatenate(phis)
        e = jnp.concatenate(es) 
        jv = jnp.concatenate(jvs)
        state = jnp.concatenate((q, phi, e, jv))

        # Jacobian-type approach
        def get_u_hats(lamb):
            u_hats = []
            for i in range(self.num_subsystems):
                if i == self.system_k_idx:
                    # exclude ground node when calculating coupling constraint (that's why es starts at 1:)
                    coupling_constraint = jnp.sum(
                        jnp.array([jnp.matmul(Al_i.T, e_i) for Al_i, e_i in zip(self.Alambdas, es[1:])])
                    )

                    u_hats.append(jnp.array([coupling_constraint]))
                else:
                    u_hats.append(-lamb)
            return u_hats

        def H_from_state(x):
            """Returns the total Hamiltonian of the composite system and finds the next state for all subsystems"""

            # Get initial u_hat
            lamb = x[-self.num_lamb:]
            u_hats = get_u_hats(lamb)
            full_states = []
            # Extract the subsystem states from the composite system state
            for i in range(self.num_subsystems):
                q_i = x[sum(self.num_capacitors[:i]) : sum(self.num_capacitors[:i+1])]
                phi_i = x[
                    nc+sum(self.num_inductors[:i]) : 
                    nc+sum(self.num_inductors[:i+1])
                ]
                e_i = x[
                    nc+nl+sum(self.num_nodes[:i]) - (i) + 1 : # + 1 to exclude ground node
                    nc+nl+sum(self.num_nodes[:i+1]) - (i) : 
                ]
                jv_i = x[
                    nc+nl+ne+sum(self.num_volt_sources[:i]) : 
                    nc+nl+ne+sum(self.num_volt_sources[:i+1])
                ] 
                e_i = jnp.concatenate((jnp.array([0]), e_i)) # append ground node
                full_state_i = jnp.concatenate((q_i, phi_i, e_i, jv_i))
                full_states.append(full_state_i)
            
            next_graphs = []
            # Get Hamiltonian predictions for all subsystems.
            # For all of the (k-1) subsystems, also get the next subsystem state.
            for i in range(self.num_subsystems):
                full_state_i = full_states[i]
                H_i, next_graph_i = H_from_state_i(full_state_i, i)

                if i != self.system_k_idx:
                    x_i = full_state_i[self.diff_indices[i]]
                    next_y_i = self.alg_vars_from_graph[i](next_graph_i)
                    next_x_i = self.ode_integrator(
                        partial(neural_dae_i, params=(i, next_y_i, u_hats)), x_i, t, self.dt, self.T
                    )
                    next_state_i = jnp.zeros((self.state_dims[i]))
                    next_state_i = next_state_i.at[self.diff_indices[i]].set(next_x_i)
                    next_state_i = next_state_i.at[self.alg_indices[i]].set(next_y_i)
                    next_graph_i = self.state_to_graph[i](next_state_i, controls[i], globals=next_graph_i.globals)
                
                next_graphs.append(next_graph_i)
            
            H = sum([next_graph.globals[0] for next_graph in next_graphs])
            return H, next_graphs
        
        def H_from_state_i(full_state_i, i):
            """Returns the Hamiltonian prediction H_i and graph_i for subsystem i"""
            if i == self.system_k_idx:
                # For the known subsystem k
                phi = full_state_i[0]
                H_k = phi
                graph_k = jraph.GraphsTuple(nodes=None,
                                            edges=None,
                                            globals=jnp.array([H_k]),
                                            receivers=None,
                                            senders=None,
                                            n_node=jnp.array([0]),
                                            n_edge=jnp.array([0]))
                return H_k, graph_k
            else:
                graph_i = self.state_to_graph[i](state=full_state_i, control=controls[i])
                intermediate_graph_i = self.train_states[i].apply_fn(
                    self.train_states[i].params, graph_i, controls[i], t, rng
                )
                H_i = intermediate_graph_i.globals[0]
                return H_i, intermediate_graph_i

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
                lamb = state[-self.num_lamb : ]
                z = jnp.concatenate((e, dHphi, dHq, jv, lamb))
                # jax.debug.print('r(z): {}', self.r(z))
                return jnp.matmul(self.J, z) - self.r(z) + jnp.matmul(self.B_bar, control)

            def monolithic_f(x, y, t, params):
                state = jnp.zeros((self.state_dim_c))
                state = state.at[self.diff_indices_c].set(x)
                state = state.at[self.alg_indices_c].set(y)
                monolithic_daes = monolithic_dynamics_function(state, t)
                # return self.U_inv @ (self.L_inv @ self.P_inv @ monolithic_daes)[self.diff_indices_c]
                return monolithic_daes[self.diff_indices_c]
            
            def monolithic_g(x, y, t, params):
                state = jnp.zeros((self.state_dim_c))
                state = state.at[self.diff_indices_c].set(x)
                state = state.at[self.alg_indices_c].set(y)
                monolithic_daes = monolithic_dynamics_function(state, t)
                return monolithic_daes[self.alg_indices_c]

            lamb0 = jnp.zeros((self.num_lamb)) # This is initial guess for lambda
            x0 = jnp.concatenate((q, phi))
            y0 = jnp.concatenate((e, jv))
            z0 = jnp.concatenate((x0, y0, lamb0))
            # TODO: try to remove all ground nodes!!!!
            # TODO: BTW, if this works every iteration, then we can just get next_state from here...
            full_system_solver = DAESolver(
                monolithic_f, monolithic_g, self.diff_indices_c, self.alg_indices_c
                )
            # Residual of subsystem GNS is already 7
            next_state = full_system_solver.solve_dae(z0, jnp.array([self.dt]), params=None, y0_tol=10)[0]
            # next_state = full_system_solver.solve_dae_one_timestep_rk4(z0, t, self.dt, params=None)
            lamb = next_state[-self.num_lamb:]
        
        u_hats = get_u_hats(lamb)
        
        def fi(x, y, t, params):
            i, u_hats = params
            state = jnp.zeros((self.state_dims[i]))
            state = state.at[self.diff_indices[i]].set(x)
            state = state.at[self.alg_indices[i]].set(y)
            dae_i = dynamics_function_i(state, t, params)
            if i == self.system_k_idx:
                return dae_i[self.diff_indices[i]]
            else:
                return self.U_invs[i] @ (self.L_invs[i] @ self.P_invs[i] @ dae_i)[self.diff_indices[i]]
        
        def gi(x, y, t, params):
            i, u_hats = params
            state = jnp.zeros((self.state_dims[i]))
            state = state.at[self.diff_indices[i]].set(x)
            state = state.at[self.alg_indices[i]].set(y)
            residuals_i = dynamics_function_i(state, t, params)[self.alg_indices[i]]
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

        # TODO: update algebraic indices at system_k_idx to include lamb!
        # TODO: try to make PH system matrices for system k with incidence matrices w/o gnd node

        k = self.system_k_idx
        dae_solver = DAESolver(
            fi, gi, self.diff_indices[k], self.alg_indices[k]
            )
        state_k_ext = jnp.concatenate((states[k], lamb))

        xk = state_k_ext[self.diff_indices[k]]
        y = state_k_ext[self.alg_indices[k]]
        yk = y[:-self.num_lamb]
        params=(k, u_hats)
        # print('f:', fi(xk, y, t, params))
        # print('g:', gi(xk, y, t, params))

        # TODO: write out equations to make sure it makes sense
        

        yknew, infodict, ier, mesg = fsolve(lambda yy : dae_solver.g(x0, yy, t, params), y0, full_output=True)

        if ier != 1:
            # throw an error if the algebraic states are not consistent.
            raise ValueError("Initial algebraic states were inconsistent. fsolve returned {}".format(mesg))
        
        state_k_ext = state_k_ext.at[self.alg_indices[k]].set(yknew)

        # next_state_k_ext = dae_solver.solve_dae(state_k_ext, jnp.array([self.dt]), params=params)
        next_state_k_ext = dae_solver.solve_dae_one_timestep_rk4(state_k_ext, t, self.dt, params)

        next_lamb = next_state_k_ext[-self.num_lamb:]

        next_state_k = next_state_k_ext[:-self.num_lamb]

        next_states.insert(k, next_state_k)
        next_graphs.insert(
            k,
            self.state_to_graph[k](
                next_state_k, controls[k], set_ground_and_control=True,
                globals=next_graphs[self.system_k_idx].globals
                )
        )

        return jraph.batch(next_graphs), next_lamb