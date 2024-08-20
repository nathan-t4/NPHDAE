import jraph
import jax
import flax.linen as nn
import jax.numpy as jnp
from typing import Callable, Union
from flax.typing import Array
from helpers.integrator_factory import integrator_factory
from utils.graph_utils import *
from utils.jax_utils import *
from utils.models_utils import *
from scripts.models.mlp import *
from scripts.models.gnn import HeterogeneousGraphNetworkSimulator
from utils.gnn_utils import *
from dae_solver.index1_semi_explicit_flax import DAESolver
    
class PHGNS(nn.Module):
    # Decoder post-processor parameters
    state_to_graph: Callable
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
            graph = self.state_to_graph(state=x, control=control, system_params=False, set_nodes=False, set_ground_and_control=False, nodes=cur_nodes, globals=None)
            aux_data = None
            processed_graph = net(graph, aux_data, rng)
            if self.include_idxs is None:
                H = jnp.sum(processed_graph.edges)
            else:
                H = jnp.dot(self.include_idxs, processed_graph.edges)
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
            graph = self.state_to_graph(state=next_state, 
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

class PHGNS_NDAE(nn.Module):
    state_to_graph: Callable
    graph_to_state: Callable
    alg_vars_from_graph: Callable
    integration_method: str
    system_config: Dict
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
    
    def setup(self):
        # Below depends on different type of system
        # TODO: learn g (voltage across resistor -> current), capacitances, and inductances as parameter
        self.AC = self.system_config['AC']
        self.AL = self.system_config['AL']
        self.AR = self.system_config['AR']
        self.AV = self.system_config['AV']
        self.AI = self.system_config['AI']
        
        self.num_nodes = self.system_config['num_nodes']
        self.num_capacitors = self.system_config['num_capacitors']
        self.num_inductors = self.system_config['num_inductors']
        self.num_resistors = self.system_config['num_resistors']
        self.num_volt_sources = self.system_config['num_volt_sources']
        self.num_cur_sources = self.system_config['num_cur_sources']
        self.state_dim = self.system_config['state_dim']

        self.E = self.system_config['E']
        self.J = self.system_config['J']
        self.r = self.system_config['r'] # this is a function!
        self.B = self.system_config['B']

        self.differential_vars = self.system_config['diff_indices']
        self.algebraic_vars = self.system_config['alg_indices']

        P, L, U = jax.scipy.linalg.lu(self.E)
        self.P_inv = jnp.linalg.inv(P)
        self.L_inv = jnp.linalg.inv(L)
        U_nonzero = U[self.differential_vars][:,self.differential_vars]
        self.U_nonzero_inv = jnp.linalg.inv(U_nonzero)

        edge_output_size = 1
        node_output_size = 1
        num_edge_types = 5 # num of two-terminal components (C, R, L, V, I)
        num_node_types = 1 # num of node types (voltage)

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
    
    def __call__(self, graph, control, t, rng):
        state = self.graph_to_state(graph)
        # Add noise to training
        if self.training: 
            edges_shape = graph.edges.shape
            rng, edges_rng = jax.random.split(rng)
            edges_noise = self.noise_std * jax.random.normal(edges_rng, edges_shape)
            noisy_edges = graph.edges + edges_noise
            new_edges = jnp.array(noisy_edges).reshape(edges_shape)
            graph = graph._replace(edges=new_edges)

        def H_from_state(x):
            graph = self.state_to_graph(state=x, control=control)
            processed_graph = self.net(graph, t, rng)
            if self.include_idxs is None:
                H = jnp.sum(processed_graph.edges)
            else:
                H = jnp.dot(self.include_idxs, processed_graph.edges).squeeze()
            return H, processed_graph
        
        def decoder_postprocessor(cur_state, control):
            H, processed_graph = H_from_state(cur_state)
            next_y = self.alg_vars_from_graph(processed_graph, self.algebraic_vars)

            def dynamics_function(z, t):
                return jnp.matmul(self.J, z) - self.r(z) + jnp.matmul(self.B, control)
            
            def f(x, t):
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
                z = jnp.zeros((self.state_dim))
                z = z.at[self.differential_vars].set(x)
                z = z.at[self.algebraic_vars].set(next_y)
                # z = jnp.concatenate((x, next_y))
                # differential_eqs = self.U_nonzero_inv @ (self.L_inv @ self.P_inv @ dynamics_function(z, t))[self.differential_vars]
                differential_eqs = dynamics_function(z, t)[self.differential_vars]
                return differential_eqs
            
            def get_residuals(x, t):
                g = dynamics_function(x, t)
                residuals = jnp.abs(g[self.algebraic_vars])
                return residuals

            if self.integration_method == 'adam_bashforth':
                integrator = integrator_factory(self.integration_method)
                cur_x = cur_state[self.differential_vars]
                next_x = integrator(f, cur_x, t, self.dt, self.T)
                next_state = jnp.zeros((self.state_dim))
                next_state = next_state.at[self.differential_vars].set(next_x)
                next_state = next_state.at[self.algebraic_vars].set(next_y)
                # next_state = jnp.concatenate((next_x, next_y))
            elif self.integration_method == 'dae':
                # Cannot use during training because Hamiltonian is incorrect
                def ff(x, y, t, params):
                    z = jnp.concatenate((x, y))
                    f = self.U_nonzero_inv @ (self.L_inv @ self.P_inv @ dynamics_function(z, t))[self.differential_vars]
                    return f
            
                def gg(x, y, t, params):
                    z = jnp.concatenate((x, y))
                    g = dynamics_function(z, t)[self.algebraic_vars]
                    return g
                
                solver = DAESolver(ff, gg, self.differential_vars, self.algebraic_vars)
                next_state = solver.solve_dae_one_timestep_rk4(cur_state, t, self.dt, params=None)
                
            residuals = get_residuals(next_state, t)
            next_globals = jnp.concatenate((jnp.array([H]), residuals))
            graph = self.state_to_graph(state=next_state, 
                                          control=control, 
                                          set_ground_and_control=True,
                                          globals=next_globals)
            
            return graph

        processed_graph = decoder_postprocessor(state, control)
        
        return processed_graph