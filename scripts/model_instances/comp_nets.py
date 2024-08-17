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
                    jnp.zeros((1, self.num_lamb)), # ground node
                    self.Alambdas[i], 
                    jnp.zeros((self.num_inductors[i]+self.num_capacitors[i]+self.num_volt_sources[i], self.num_lamb))
                ))
                self.B_hats = [*self.B_hats, B_hat_i]
            
        assert(self.system_k_idx != -1), \
            "k-th system index has not been set. \
            Make sure that one system config in self.system_configs has cfg['is_k'] = True!"
        
        num_capacitors_c = sum(self.num_capacitors)
        num_resistors_c = sum(self.num_resistors)
        num_inductors_c = sum(self.num_inductors)
        num_volt_sources_c = sum(self.num_volt_sources)
        num_cur_sources_c = sum(self.num_cur_sources)
        # TODO: avoid recounting gnd
        num_nodes_c = sum(self.num_nodes) # - self.num_subsystems + 1 
        state_dim_c = sum(self.num_capacitors) + sum(self.num_inductors) + sum(self.num_nodes) + sum(self.num_volt_sources)

        self.diff_indices_c = jnp.arange(num_capacitors_c + num_inductors_c)
        self.alg_indices_c = jnp.arange(
            num_capacitors_c + num_inductors_c,
            num_capacitors_c + num_inductors_c + num_nodes_c + num_volt_sources_c 
        )
        self.num_diff_vars_c = sum(self.num_diff_vars)
        self.num_alg_vars_c = sum(self.num_alg_vars)

        # keep track of nonzero columns
        AC_c = []; AR_c = []; AL_c = []; AV_c = []; AI_c = []
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

        comp_AC = jax.scipy.linalg.block_diag(*self.ACs)[:,AC_c]
        comp_AR = jax.scipy.linalg.block_diag(*self.ARs)[:,AR_c]
        comp_AL = jax.scipy.linalg.block_diag(*self.ALs)[:,AL_c]
        comp_AV = jax.scipy.linalg.block_diag(*self.AVs)[:,AV_c]
        comp_AI = jax.scipy.linalg.block_diag(*self.AIs)[:,AI_c]
        
        # TODO: composite incidence matrices need to ignore all zero columns (present in the transmission line)...
        composite_system_config = {
            'AC': comp_AC,
            'AR': comp_AR,
            'AL': comp_AL,
            'AV': comp_AV,
            'AI': comp_AI,
            'Alambda': self.Alambda,
            'num_capacitors': num_capacitors_c,
            'num_inductors': num_inductors_c,
            'num_nodes': num_nodes_c,
            'num_volt_sources': num_volt_sources_c,
            'num_cur_sources': num_cur_sources_c,
            'num_lamb': self.num_lamb,
            'state_dim': state_dim_c,
            'diff_indices': self.diff_indices,
            'alg_indices': self.alg_indices,
            'num_diff_vars': self.num_diff_vars_c,
            'num_alg_vars': self.num_alg_vars_c,
        }

        # Equation (15) - I think there is typo on equation (18)
        self.J = get_J_matrix(composite_system_config)
        self.E = get_E_matrix(composite_system_config)

        def get_composite_r(z):
            g = lambda e : (comp_AR.T @ e) / 1.0 
            # z = [e, jl, uc, jv, lamb]
            e = z[0 : num_nodes_c]
            jl = z[num_nodes_c: num_nodes_c+num_inductors_c]
            uc = z[
                num_nodes_c+num_inductors_c : 
                num_nodes_c+num_inductors_c+num_capacitors_c
                ]
            jv = z[
                num_nodes_c+num_inductors_c+num_capacitors_c :
                num_nodes_c+num_inductors_c+num_capacitors_c+num_volt_sources_c
                ]
            lamb = z[-self.num_lamb:]

            curr_through_resistors = jnp.linalg.matmul(comp_AR, g(e))
            charge_constraint = jnp.matmul(comp_AC.T, e) - uc

            diss = jnp.zeros((state_dim_c,))
            diss = diss.at[0:num_nodes_c].set(curr_through_resistors)
            diss = diss.at[(num_nodes_c+num_capacitors_c):(num_nodes_c+num_capacitors_c+ num_capacitors_c)].set(charge_constraint)

            return diss
        
        self.r = get_composite_r
        self.B_bar = get_B_bar_matrix(composite_system_config)
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
        qs = []
        phis = []
        es = []
        jvs = []
        for i, state in enumerate(states):
            qs.append(state[0 : self.num_capacitors[i]])
            phis.append(state[self.num_capacitors[i] : self.num_capacitors[i]+self.num_inductors[i]])
            # TODO: what about the ground node?
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

        def H_from_state(x):
            # Step 2
            nc = sum(self.num_capacitors)
            nl = sum(self.num_inductors)
            ne = sum(self.num_nodes)
            full_states = []
            # Solve the first k-1 subsystems
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
                full_state_i = jnp.concatenate((q_i, phi_i, e_i, jv_i))
                full_states.append(full_state_i)
            
            next_graphs = []
            dH = []
            for i in range(self.num_subsystems):
                full_state_i = full_states[i]
                if i == self.system_k_idx:
                    # Solve the k-th subsystem
                    def H_from_state_k(full_state_k):
                        phi = full_state_k[0]
                        return phi
                    
                    # system_k_solver = DAESolver(fi, gi, self.num_diff_vars[i], self.num_alg_vars[i]+self.num_lamb)

                    H_k, dH_k = jax.value_and_grad(H_from_state_k)(full_state_i)
                    graph_k = jraph.GraphsTuple(nodes=None,
                                                edges=None,
                                                globals=jnp.array([H_k]),
                                                receivers=None,
                                                senders=None,
                                                n_node=jnp.array([0]),
                                                n_edge=jnp.array([0]))
                    next_graphs.append(graph_k)
                    dH.append(dH_k)
                
                else:
                    # Solve the (k-1) subsystems
                    def H_from_state_i(full_state_i):
                        graph_i = self.state_to_graph[i](state=full_state_i, control=controls[i])
                        intermediate_graph_i = self.train_states[i].apply_fn(
                            self.train_states[i].params, graph_i, controls[i], t, rng
                        )
                        H_i = intermediate_graph_i.globals[0]
                        return H_i, intermediate_graph_i
                    
                    dH_i, intermediate_graph_i = jax.grad(H_from_state_i, has_aux=True)(full_state_i)
                    x_i = full_state_i[self.diff_indices[i]]
                    next_y_i = self.alg_vars_from_graph[i](intermediate_graph_i)
                    next_x_i = self.ode_integrator(
                        partial(neural_dae_i, params=(i, next_y_i, u_hats, dH_i)), x_i, t, self.dt, self.T
                    )
                    next_state_i = jnp.concatenate((next_x_i, next_y_i))
                    next_graph_i = self.state_to_graph[i](next_state_i, controls[i], globals=intermediate_graph_i.globals)
                    dH.append(dH_i)
                    next_graphs.append(next_graph_i)
            
            H = sum([next_graph.globals[0] for next_graph in next_graphs])
            return H, dH, next_graphs

        def dynamics_function_i(x, t, params):
            i, u_hats, dH_i = params
            e_i = x[
                self.num_capacitors[i]+self.num_inductors[i] : 
                self.num_capacitors[i]+self.num_inductors[i]+self.num_nodes[i]
            ]
            jv_i = x[
                self.num_capacitors[i]+self.num_inductors[i]+self.num_nodes[i] : 
                self.num_capacitors[i]+self.num_inductors[i]+self.num_nodes[i]+self.num_volt_sources[i]
            ]
            dHq = dH_i[:self.num_capacitors[i]]
            dHphi = dH_i[self.num_capacitors[i] : self.num_capacitors[i]+self.num_inductors[i]]

            if i == self.system_k_idx:
                lamb = x[-self.num_lamb:]
                z_i = jnp.concatenate((e_i, dHphi, dHq, jv_i, lamb))
            else:
                z_i = jnp.concatenate((e_i, dHphi, dHq, jv_i))

            return jnp.matmul(self.Js[i], z_i) - self.rs[i](z_i) + jnp.matmul(self.B_bars[i], controls[i]) + jnp.matmul(self.B_hats[i], u_hats[i])
        
        def neural_dae_i(x, t, params):
            i, next_y, u_hats, dH_i = params
            z = jnp.concatenate((x, next_y))
            daes = dynamics_function_i(z, t, (i, u_hats, dH_i))
            return self.U_invs[i] @ (self.L_invs[i] @ self.P_invs[i] @ daes)[self.diff_indices[i]]
        
        # Step 1: solve for lambda_0 by solving full system on the first iteration
        # TODO: the problem with this approach is that the Hamiltonian of the k-th subsystem is not included...
        # TODO: J for composite system looks a bit different...
        u_hats = []
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
                lamb = state[-self.num_lamb : ]
                z = jnp.stack((e, dHphi, dHq, jv, lamb))
                return jnp.matmul(self.J, z) - self.r(z) + jnp.matmul(self.B_bar, control)

            def monolithic_f(x, y, t, params):
                monolithic_daes = monolithic_dynamics_function(jnp.concatenate((x, y)), t)
                return self.U_inv @ (self.L_inv @ self.P_inv @ monolithic_daes)[self.diff_indices_c]
            
            def monolithic_g(x, y, t, params):
                monolithic_daes = monolithic_dynamics_function(jnp.concatenate((x, y)), t)
                return monolithic_daes[self.alg_indices_c]
            
            lamb = jnp.zeros((self.num_lamb)) # This is initial guess for lambda
            state_ext = jnp.concatenate((q, phi, e, jv, lamb))
            # TODO: BTW, if this works every iteration, then we can just get next_state from here...
            full_system_solver = DAESolver(monolithic_f, monolithic_g, self.num_diff_vars_c, self.num_alg_vars_c+self.num_lamb)
            next_state = full_system_solver.solve_dae_one_timestep_rk4(state_ext, t, self.dt, params=None)
            lamb = next_state[-self.num_lamb:]
        
        # Jacobian-type approach
        for i in range(self.num_subsystems):
            if i == self.system_k_idx:
                # TODO: appended zeros, but maybe Al_i shouldn't be zeros...
                coupling_constraint = jnp.sum(jnp.array([
                        jnp.matmul(jnp.concatenate(
                            (jnp.zeros((1, Al_i.shape[1])), Al_i)).T, e_i) 
                        for Al_i, e_i in zip(self.Alambdas, es)
                        ])
                )
                u_hats.append(jnp.array([coupling_constraint]))
            else:
                u_hats.append(-lamb)
            
        
        def fi(x, y, t, params):
            i, u_hats, dH_i = params
            dae_i = dynamics_function_i(jnp.concatenate((x, y)), t, params)
            if len(self.diff_indices[i]) == 1:
                return dae_i[self.diff_indices[i]]
            else:
                return self.U_invs[i] @ (self.L_invs[i] @ self.P_invs[i] @ dae_i)[self.diff_indices[i]]
        
        def gi(x, y, t, params):
            i, u_hats, dH_i = params
            residuals_i = dynamics_function_i(jnp.concatenate((x, y)), t, params)[self.alg_indices[i]]
            return residuals_i
        

        H, dH, next_graphs = H_from_state(state)
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
        k = self.system_k_idx
        dae_solver = DAESolver(
            fi, gi, self.num_diff_vars[k], self.num_alg_vars[k]
            )
        state_k_ext = jnp.concatenate((states[k], lamb))

        x = state_k_ext[self.diff_indices[k]]
        y = state_k_ext[self.alg_indices[k]]
        params=(k, u_hats, dH[k])
        print('f:', fi(x, y, t, params))
        print('g:', gi(x, y, t, params))


        next_state_k_ext = dae_solver.solve_dae_one_timestep_rk4(
            state_k_ext, t, self.dt, 
            params=(k, u_hats, dH[k])
            )
        next_lamb = next_state_k_ext[-self.num_lamb:]

        next_state_k = next_state_k_ext[:-self.num_lamb]
        next_states.insert(k, next_state_k)
        next_graphs.insert(
            k,
            self.graph_to_state[k](
                next_state_k, controls[k], # set_ground_and_control=True,
                globals=next_graphs[self.system_k_idx].globals
                )
        )

        return jraph.batch(next_graphs), next_lamb