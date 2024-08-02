import jraph
import jax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from typing import Callable, Union
from flax.training.train_state import TrainState
from flax.typing import Array
from ml_collections import FrozenConfigDict
from helpers.integrator_factory import integrator_factory
from utils.graph_utils import *
from utils.jax_utils import *
from utils.models_utils import *
from scripts.models import MLP, vmapMLP, GraphNetworkSimulator, HeterogeneousGraphNetworkSimulator
from scipy_dae.integrate import solve_dae
from diffrax import diffeqsolve, ODETerm, ImplicitEuler
import scipy
import time
from utils.gnn_utils import *

class MassSpringGNS(nn.Module):
    # Decoder post-processor parameters
    norm_stats: FrozenConfigDict
    integration_method: str = 'SemiImplicitEuler'
    dt: float = 0.01 # TODO: set from graphbuilder?

    # Graph Network parameters
    num_mp_steps: int = 1
    use_edge_model: bool = False
    shared_params: bool = False
    globals_output_size: int = 0
    edge_output_size: int = 1
    node_output_size: int = 1
    
    # Encoder/Decoder MLP parameters
    layer_norm: bool = False
    latent_size: int = 16
    hidden_layers: int = 2
    activation: str = 'relu'
    dropout_rate: float = 0
    training: bool = True

    # Graph parameters
    add_self_loops: bool = False
    add_undirected_edges: bool = False
    vel_history: int = 1
    control_history: int = 1
    noise_std: float = 0.0003

    def setup(self):
        encoder_node_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers,
                              with_layer_norm=self.layer_norm, 
                              activation=self.activation)
        encoder_edge_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers,
                              with_layer_norm=self.layer_norm, 
                              activation=self.activation)

        decoder_node_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers + [self.node_output_size],
                              activation=self.activation)
        decoder_edge_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers + [self.edge_output_size], 
                              activation=self.activation)

        def decoder_postprocessor(graph: jraph.GraphsTuple, aux_data):
            cur_pos, cur_vel, prev_vel, prev_control, num_nodes = aux_data
            next_nodes = None
            next_edges = None

            integrator = MassSpringIntegrator(self.dt, self.num_mp_steps, self.norm_stats, self.integration_method)
            cur_state = jnp.concatenate((cur_pos, cur_vel))
            next_pos, next_vel, prediction = integrator.dynamics_function(cur_state, 0.0, graph)
            next_nodes = jnp.column_stack((next_pos, 
                                        prev_vel[:,1:], next_vel, 
                                        prev_control[:,1:], 
                                        prediction))
            next_edges = jnp.diff(next_pos.squeeze()).reshape(-1,1)
            
            if self.add_undirected_edges:
                next_edges = jnp.concatenate((next_edges, next_edges), axis=0)
            
            if self.add_self_loops:
                next_edges = jnp.concatenate((next_edges, jnp.zeros((num_nodes, 1))), axis=0)
            
            if self.use_edge_model:
                graph = graph._replace(nodes=next_nodes, edges=next_edges)
            else:
                graph = graph._replace(nodes=next_nodes)   

            return graph

        self.net = GraphNetworkSimulator(
            encoder_node_fn=encoder_node_fn,
            encoder_edge_fn=encoder_edge_fn,
            decoder_node_fn=decoder_node_fn,
            decoder_edge_fn=decoder_edge_fn,
            decoder_postprocessor=decoder_postprocessor,
            num_mp_steps=self.num_mp_steps,
            shared_params=self.shared_params,
            use_edge_model=self.use_edge_model,
            latent_size=self.latent_size,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            training=self.training,
            layer_norm=self.layer_norm,
        )
    
    def __call__(self, graph, control, rng):
        num_nodes = len(graph.nodes)
        pos = graph.nodes[:,0]
        control = control[1::2] # get nonzero elements (even indices) corresponding to control input
        if self.training: 
            # Add noise to current position (first node feature)
            rng, pos_rng, u_rng = jax.random.split(rng, 3)
            pos_noise = self.noise_std * jax.random.normal(pos_rng, (num_nodes,))
            pos = pos + pos_noise
            # Add noise to control input at current time-step (next_u) TODO: was commented out
            control_noise = self.noise_std * jax.random.normal(u_rng, (num_nodes,))
            control = control + control_noise

        new_nodes = jnp.column_stack((pos, graph.nodes[:,1:], control))
        graph = graph._replace(nodes=new_nodes)

        cur_pos = graph.nodes[:,0]
        cur_vel = graph.nodes[:,self.vel_history]
        prev_vel = graph.nodes[:,1:self.vel_history+1] # includes current velocity
        prev_control = graph.nodes[:,self.vel_history+1:] # includes current u

        aux_data = (cur_pos, cur_vel, prev_vel, prev_control, num_nodes)

        return self.net(graph, aux_data, rng)
    
class PHGNS(nn.Module):
    # Decoder post-processor parameters
    graph_from_state: Callable
    J: Union[None, Array] # if None then learn
    R: Union[None, Array] # if None then learn
    g: Union[None, Array] # if None then learn

    integration_method: str
    dt: float
    T: int

    # Graph Network parameters
    edge_idxs: Array
    node_idxs: Array
    include_idxs: Array
    num_mp_steps: int
    learn_nodes: bool
    use_edge_model: bool
    use_global_model: bool
    shared_params: bool
    
    # Encoder/Decoder MLP parameters
    layer_norm: bool
    latent_size: int
    hidden_layers: int
    activation: str 
    dropout_rate: float
    training: bool

    # Graph parameters
    noise_std: float

    @nn.compact
    def __call__(self, graph, control, rng):
        '''
            TODO:
            - encoder_node_fn -> encoder_node_fns
        '''
        edge_output_size = 1
        node_output_size = 1
        num_edge_types = 1 if self.edge_idxs is None else (len(self.edge_idxs) + 1)
        num_node_types = 1 if self.node_idxs is None else (len(self.node_idxs) + 1)
        # encoder_node_fn = vmapMLP(feature_sizes=[self.latent_size] * (self.hidden_layers),
        #                           with_layer_norm=self.layer_norm, 
        #                           activation=self.activation, 
        #                           name='enc_node') # TODO: GraphMapFeatures only batches edges not nodes... 

        encoder_edge_fns = [MLP(feature_sizes=[self.latent_size] * self.hidden_layers,
                                with_layer_norm=self.layer_norm, 
                                activation=self.activation, 
                                name=f'enc_edge_{i}') for i in range(num_edge_types)]
        
        encoder_node_fns = [MLP(feature_sizes=[self.latent_size] * self.hidden_layers,
                                with_layer_norm=self.layer_norm, 
                                activation=self.activation, 
                                name=f'enc_node_{i}') for i in range(num_node_types)]

        # decoder_node_fn = vmapMLP(feature_sizes=[self.latent_size] * self.hidden_layers + [node_output_size],
        #                           activation=self.activation, 
        #                           name='dec_node')
        
        decoder_edge_fns = [MLP(feature_sizes=[self.latent_size] * self.hidden_layers + [edge_output_size], 
                                activation=self.activation, 
                                name=f'dec_edge_{i}') for i in range(num_edge_types)]
        decoder_node_fns = [MLP(feature_sizes=[self.latent_size] * self.hidden_layers + [node_output_size], 
                                activation=self.activation, 
                                name=f'dec_node_{i}') for i in range(num_node_types)]

        net = HeterogeneousGraphNetworkSimulator(
            edge_idxs=self.edge_idxs,
            node_idxs=self.node_idxs,
            encoder_node_fns=encoder_node_fns,
            encoder_edge_fns=encoder_edge_fns,
            decoder_node_fns=decoder_node_fns,
            decoder_edge_fns=decoder_edge_fns,
            decoder_postprocessor=lambda x, _: x, # identity decoder post-processor
            num_mp_steps=self.num_mp_steps,
            shared_params=self.shared_params,
            learn_nodes=self.learn_nodes,
            use_edge_model=self.use_edge_model,
            use_global_model=self.use_global_model,
            dt=self.dt,
            T=self.T,
            latent_size=self.latent_size,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            training=self.training,
            layer_norm=self.layer_norm,
            name='GNN',
        )
        edges_shape = graph.edges.shape
        cur_nodes = graph.nodes
        state = graph.edges[:,0].squeeze()
        if self.training: 
            rng, edges_rng = jax.random.split(rng)
            edges_noise = self.noise_std * jax.random.normal(edges_rng, edges_shape)
            noisy_edges = graph.edges + edges_noise
            new_edges = jnp.array(noisy_edges).reshape(edges_shape)
            graph = graph._replace(edges=new_edges)

        def H_from_state(x):
            graph = self.graph_from_state(state=x, control=control, system_params=False, set_nodes=False, set_ground_and_control=False, nodes=cur_nodes, globals=None)
            aux_data = None
            processed_graph = net(graph, aux_data, rng)
            if self.include_idxs is None:
                H = jnp.sum(processed_graph.edges)
            else:
                H = jnp.sum(processed_graph.edges[self.include_idxs])
            return H, processed_graph
        
        H, processed_graph = H_from_state(state)

        def get_learned_J():
            out_dim = len(state)
            J = nn.Dense(features=out_dim, use_bias=False, name='J')(jnp.eye(out_dim))
            J_triu = jnp.triu(J)
            return J_triu - J_triu.T  # Make J skew-symmetric
        
        def get_learned_R():
            out_dim = len(state)
            L = nn.Dense(features=out_dim, use_bias=False, name='R')(jnp.eye(out_dim))
            L_tril = jnp.tril(L)
            return jnp.matmul(L_tril, L_tril.T)  # Make R positive-definite and symmetric
                
        def get_learned_g():
            out_dim = len(state)
            g = nn.Dense(features=out_dim, use_bias=False, name='g')(jnp.eye(out_dim))
            return g

        J = self.J if self.J is not None else get_learned_J()
        R = self.R if self.R is not None else get_learned_R()
        g = self.g if self.g is not None else get_learned_g()

        def decoder_postprocessor(graph: jraph.GraphsTuple, aux_data):
            H, cur_state = aux_data
            def dynamics_function(x, t):                
                dH, _ = jax.grad(H_from_state, has_aux=True)(x)
                z = dH
                return jnp.matmul(J - R, z).squeeze() + jnp.matmul(g, control).squeeze()
            
            if self.integration_method == 'adam_bashforth': # The only multi-step method implemented
                next_state = integrator_factory(self.integration_method)(dynamics_function, cur_state, 0.0, self.dt, self.T)
            else: # single-step integration scheme
                next_state = integrator_factory(self.integration_method)(dynamics_function, cur_state, 0.0, self.dt)
            
            next_globals = jnp.array(H)
            graph = self.graph_from_state(state=next_state, 
                                          control=control, 
                                          system_params=False, 
                                          set_nodes=False,
                                          set_ground_and_control=True, 
                                          nodes=graph.nodes, 
                                          globals=next_globals)
            return graph

        aux_data = H, state
        processed_graph = decoder_postprocessor(processed_graph, aux_data)
        
        return processed_graph

class PHGNS2(nn.Module):
    # Decoder post-processor parameters
    graph_from_state: Callable
    integration_method: str
    dt: float
    T: int

    # Graph Network parameters
    edge_idxs: Array
    node_idxs: Array
    include_idxs: Array
    incidence_matrices: Array
    splits: Array
    num_mp_steps: int
    learn_nodes: bool
    use_edge_model: bool
    use_global_model: bool
    shared_params: bool
    
    # Encoder/Decoder MLP parameters
    layer_norm: bool
    latent_size: int
    hidden_layers: int
    activation: str 
    dropout_rate: float
    training: bool

    # Graph parameters
    noise_std: float
    
    def setup(self):
        # Below depends on different type of system
        AC, AR, AL, AV, AI = self.incidence_matrices
        Nnode = len(AC) # TODO: What if AC is None?
        Nq = 0 if AC is None else len(AC.T)
        Nphi = 0 if AL is None else len(AL.T)
        Ne = Nnode 
        Nv = 0 if AV is None else len(AV.T)

        E = jnp.block([[AC, jnp.zeros((Nnode, Nphi)), jnp.zeros((Nnode, Ne))],
                        [jnp.zeros((Nphi, Nq)), jnp.eye(Nphi), jnp.zeros((Nphi, Ne))],
                        [jnp.zeros((1, Nq)), jnp.zeros((1, Nphi)), jnp.zeros((1, Ne))],
                        [jnp.zeros((1, Nq)), jnp.zeros((1, Nphi)), jnp.zeros((1, Ne))]])
        
        if Nv != 0:
            extra_col = jnp.r_[jnp.zeros((Nq, Nv)), jnp.zeros((Nphi, Nv)), jnp.zeros((1, Nv)), jnp.zeros(1, Nv)]
            E = jnp.concatenate((E, extra_col), axis=1)
            
        # J = jnp.block([[jnp.zeros((Ne, Ne)), -AL, jnp.zeros((Ne,)), -AV],
        #                [AL.T, jnp.zeros((Nphi, Nphi)), jnp.zeros((Nphi,)), jnp.zeros()],
        #                [jnp.zeros(), jnp.zeros(), jnp.zeros(), jnp.zeros()],
        #                [AV.T, jnp.zeros(), jnp.zeros(), jnp.zeros()]])
        # R = jnp.block()
        # g = jnp.block([[-AI, jnp.zeros()],
        #                [jnp.zeros(), jnp.zeros()],
        #                [jnp.zeros(), jnp.zeros()],
        #                [jnp.zeros(), -jnp.eye()]])

        P, L, U = jax.scipy.linalg.lu(E)
        self.P_inv = jnp.linalg.inv(P)
        self.L_inv = jnp.linalg.inv(L)
        # self.differential_vars = get_nonzero_row_indices(U)
        # self.algebraic_vars = get_zero_row_indices(U)
        self.differential_vars = jnp.array([0, 1, 2]) # TODO: automate
        self.algebraic_vars = jnp.array([3, 4, 5]) # TODO: automate
        U_nonzero = U[self.differential_vars][:,self.differential_vars]
        self.U_nonzero_inv = jnp.linalg.inv(U_nonzero)

        edge_output_size = 1
        node_output_size = 1
        num_edge_types = 1 if self.edge_idxs is None else (len(self.edge_idxs) + 1)
        num_node_types = 1 if self.node_idxs is None else (len(self.node_idxs) + 1)

        encoder_edge_fns = [MLP(feature_sizes=[self.latent_size] * self.hidden_layers,
                                with_layer_norm=self.layer_norm, 
                                activation=self.activation, 
                                name=f'enc_edge_{i}') for i in range(num_edge_types)]
        
        encoder_node_fns = [MLP(feature_sizes=[self.latent_size] * self.hidden_layers,
                                with_layer_norm=self.layer_norm, 
                                activation=self.activation, 
                                name=f'enc_node_{i}') for i in range(num_node_types)]
        
        decoder_edge_fns = [MLP(feature_sizes=[self.latent_size] * self.hidden_layers + [edge_output_size], 
                                activation=self.activation, 
                                name=f'dec_edge_{i}') for i in range(num_edge_types)]
        decoder_node_fns = [MLP(feature_sizes=[self.latent_size] * self.hidden_layers + [node_output_size], 
                                activation=self.activation, 
                                name=f'dec_node_{i}') for i in range(num_node_types)]

        self.net = HeterogeneousGraphNetworkSimulator(
            edge_idxs=self.edge_idxs,
            node_idxs=self.node_idxs,
            encoder_node_fns=encoder_node_fns,
            encoder_edge_fns=encoder_edge_fns,
            decoder_node_fns=decoder_node_fns,
            decoder_edge_fns=decoder_edge_fns,
            decoder_postprocessor=lambda x, _: x, # identity decoder post-processor
            num_mp_steps=self.num_mp_steps,
            shared_params=self.shared_params,
            learn_nodes=self.learn_nodes,
            use_edge_model=self.use_edge_model,
            use_global_model=self.use_global_model,
            dt=self.dt,
            T=self.T,
            latent_size=self.latent_size,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            training=self.training,
            layer_norm=self.layer_norm,
            name='GNN',
        )

    def __call__(self, graph, control, rng):
        # TODO: what shape should control be? (num_voltage_sources + num_current_sources)
        edges_shape = graph.edges.shape
        cur_nodes = graph.nodes
        state = jnp.concatenate((graph.edges[jnp.array([0, 2]), 0], # capacitor indices
                                 graph.edges[jnp.array([1]), 0], # inductor indices
                                 graph.nodes.squeeze()) # node voltages
                               ) # TODO: no voltage source indices
        # state = self.get_state_from_graph(graph) # TODO
        if self.training: 
            rng, edges_rng = jax.random.split(rng)
            edges_noise = self.noise_std * jax.random.normal(edges_rng, edges_shape)
            noisy_edges = graph.edges + edges_noise
            new_edges = jnp.array(noisy_edges).reshape(edges_shape)
            graph = graph._replace(edges=new_edges)

        def H_from_state(x):
            graph = self.graph_from_state(state=x, control=control, system_params=False, set_nodes=False, set_ground_and_control=False, nodes=cur_nodes, globals=None)
            aux_data = None
            processed_graph = self.net(graph, aux_data, rng)
            if self.include_idxs is None: # TODO: Change indexes to only differential (energy storing) states
                H = jnp.sum(processed_graph.edges) # TODO: change sequence of edges to [q, phi]
            else:
                energy_indices = jnp.unique(jnp.concatenate((self.include_idxs, self.differential_vars)))
                H = jnp.sum(processed_graph.edges[energy_indices])
            return H, processed_graph
        
        def decoder_postprocessor(cur_state):
            H, processed_graph = H_from_state(cur_state)
            AC, AR, AL, AV, AI = self.incidence_matrices
            N_nodes = len(processed_graph.nodes)
            AC = AC if AC is not None else jnp.zeros((N_nodes, 1))
            AR = AR if AR is not None else jnp.zeros((N_nodes, 1))
            AL = AL if AL is not None else jnp.zeros((N_nodes, 1))
            AV = AV if AV is not None else jnp.zeros((N_nodes, 1))
            AI = AI if AI is not None else jnp.zeros((N_nodes, 1))
            # i, v = control
            i = jnp.zeros((1,)) # TODO: for now
            v = jnp.zeros((1,)) # TODO: for now

            g = lambda e : (AR.T @ e) / 1.0 # 1.0 is resistance R
            e_indices = self.algebraic_vars - len(self.differential_vars)
            next_y = processed_graph.nodes[e_indices].squeeze() # train weights by minimizing algebraic residual
            
            def f(x, t):
                q = x[0:self.splits[0]]
                phi = x[self.splits[0]:self.splits[1]]
                e = x[self.splits[1]: self.splits[2]]
                jv = x[self.splits[2]:len(cur_state)]
                dH, _ = jax.grad(H_from_state, has_aux=True)(x)
                dH0 = dH[jnp.arange(self.splits[0])]
                dH1 = dH[jnp.arange(self.splits[0], self.splits[1])]
                F0 = -AL @ dH1 - AI @ i - AR @ g(e) - (AV @ jv if len(jv) > 0 else 0)
                F1 = AL.T @ e
                F2 = -(AC.T @ e - dH0)
                F3 = AV.T @ e - v                    
                return jnp.stack([F0, F1, F2, F3]) if len(jv) > 0 else jnp.concatenate((F0, F1, F2))
            
            def dynamics_function(x, t): # t, x, aux_data 
                '''
                    P, L, U = jax.scipy.linalg.lu(E)
                    E = P @ L @ U
                    P @ L @ U @ x_dot = J @ z - r
                    U @ x_dot = L^{-1} @ P^{-1} @ (J @ z - r)
                    # Next only take the nonzero rows of U
                    U_nonzero = jnp.nonzero(U)
                    nonzero_indices = ... 
                    # The above only needs to be done on the first forward pass
                    x_dot = U_nonzero^{-1} @ (L^{-1} @ P^{-1} @ (J @ z - r))[nonzero_indices]
                '''
                # updated_state = np.empty(len(cur_state))
                # updated_state[differential_vars] = x
                # updated_state[algebraic_vars] = next_y
                # updated_state = jnp.array(updated_state)
                updated_state = jnp.concatenate((x, next_y))
                differential_eqs = self.U_nonzero_inv @ (self.L_inv @ self.P_inv @ (f(updated_state, t)))[self.differential_vars]
                return differential_eqs
            
            def get_residual(x, t):
                equations = f(x, t) # state_dot
                residual = jnp.abs(equations[self.algebraic_vars])
                return residual

            t = 0.0
            cur_x = cur_state[self.differential_vars]
            if self.integration_method == 'adam_bashforth':
                next_x = integrator_factory(self.integration_method)(dynamics_function, cur_x, t, self.dt, self.T)
                # next_state = np.empty(len(cur_state))
                # next_state[differential_vars] = next_x
                # next_state[algebraic_vars] = next_y
                # next_state = jnp.array(next_state)
                next_state = jnp.concatenate((next_x, next_y)) # TODO: replace
            else:
                raise NotImplementedError()
            residual = get_residual(next_state, t)
            
            next_globals = jnp.concatenate((jnp.array([H]), residual))
            graph = self.graph_from_state(state=next_state, 
                                          control=control, 
                                          system_params=False, 
                                          set_nodes=False,
                                          set_ground_and_control=True, # TODO: Was TRUE
                                          nodes=processed_graph.nodes, 
                                          globals=next_globals)
            
            return graph

        processed_graph = decoder_postprocessor(state)
        
        return processed_graph

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
    graph_to_state_one: Callable = None # TODO
    graph_to_state_two: Callable = None # TODO
    state_to_graph_one: Callable = None
    state_to_graph_two: Callable = None

    @nn.compact
    def __call__(self, graph1, graph2, rng):
        '''
            1. Estimate H1 and H2 using GNS1 and GNS2 respectively
            2. The coupled system is a PH-DAE with an additional variable lambda & incidence matrices A_lambda_i for each subsystem
                a. A_lambda_i is known (depending on where the systems are interconnected, add an artificial voltage source)
            3. Dynamic iteration scheme
                a. solve each subsystem independently using modified input. but one subsystem is augmented with extra state lambda and constraint.

            The subsystems inputs and outputs are related somehow...
            
            When composing, we add artificial voltage sources at every point where we 'merge nodes'
                - the artificial voltage sources goes between the merged node and ground
                - this basically places an additional constraint on the whole system. This constraint is placed on the node voltages 'e'
                - OR these artificial voltage sources adds multiple new constraints. One constraint per each merging
                    - e.g. e21 - e11 = 0

            Is there anyway to reuse the next algebraic state predictor (MLP) used in the forward passes of the subsystem GNNs?
                - next algebraic state predictor: current subsystem state -> next subsystem algebraic state
                - look at how traditional DAE solves work. Maybe the learned component can be more low level, and then the subsystem learned components can be connected at a lower level, so that the DAE solver can still be used after composing

            The constraints can be split into different types:
                - KCL (current_in = current_out)
                - state-voltage relations (e.g. V = Q/C)


        '''
        senders1 = graph1.senders
        receivers1 = graph1.receivers
        senders2 = graph2.senders
        receivers2 = graph2.receivers

        # cur_nodes1 = graph1.nodes
        # state1 = graph1.edges[:,0].squeeze()
        # Q1, Phi1, Q3_1 = state1

        state1 = self.graph_to_state_one(graph1)
        state2 = self.graph_to_state_two(graph2)

        # cur_nodes2 = graph2.nodes
        # state2 = graph2.edges[:,0].squeeze()
        # Q2, Phi2, Q3_2 = state2

        # full_state = jnp.concatenate((state1, state2[:2]))
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