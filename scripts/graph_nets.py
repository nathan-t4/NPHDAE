import jraph
import jax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from flax.training.train_state import TrainState
from ml_collections import FrozenConfigDict
from integrators.rk4 import rk4
from integrators.euler_variants import euler, semi_implicit_euler
from utils.graph_utils import *
from utils.jax_utils import *
from utils.models_utils import *
from scripts.models import MLP, GraphNetworkSimulator, CustomEdgeGraphNetworkSimulator

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
    
class LCGNS(nn.Module):
    # Decoder post-processor parameters
    system_params: dict
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
        decoder_edgeC_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers + [self.edge_output_size], 
                              activation=self.activation)
        decoder_edgeL_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers + [self.edge_output_size], 
                              activation=self.activation)

        self.net = CustomEdgeGraphNetworkSimulator(
            edge_idxs=[0],
            encoder_node_fn=encoder_node_fn,
            encoder_edge_fn=encoder_edge_fn,
            decoder_node_fn=decoder_node_fn,
            decoder_edge_fn_1=decoder_edgeC_fn,
            decoder_edge_fn_2=decoder_edgeL_fn,
            decoder_postprocessor=lambda x, _: x, # identity decoder postprocessor
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
        num_edges = len(graph.edges)
        cur_v = graph.nodes[:,0]
        cur_state = graph.edges[:,0]
        cur_Q, cur_Phi = cur_state
        if self.training: 
            rng, state_rng, voltage_rng = jax.random.split(rng, 3)
            state_noise = self.noise_std * jax.random.normal(state_rng, (num_edges,))
            noisy_state = cur_state + state_noise
            # TODO: add noise to voltage (nodes?)

            new_edges = jnp.array(noisy_state).reshape(-1, 1)
            graph = graph._replace(edges=new_edges)

        def H_from_state(Q, Phi):
            '''
                1. state to graph
                2. processed_graph = self.net(graph, aux_data, rng)
                3. return Hamiltonian (processed_graph.globals?)
            '''
            n_node = jnp.array([2])
            n_edge = jnp.array([2])
            senders = jnp.array([0, 1])
            receivers = jnp.array([1, 0])

            # From LCGraphBuilder
            V = Q / self.system_params['C']
            nodes = jnp.array([[0], [V]])
            edges = jnp.array([[Q], [Phi]])
            global_context = None

            graph = jraph.GraphsTuple(
                nodes=nodes,
                edges=edges,
                globals=global_context,
                n_node=n_node,
                n_edge=n_edge,
                senders=senders,
                receivers=receivers,
            )
            aux_data = None
            processed_graph = self.net(graph, aux_data, rng)
            return jnp.sum(processed_graph.edges), processed_graph
        
        H, _ = H_from_state(cur_Q, cur_Phi)
        H_grads, processed_graph = jax.grad(H_from_state, argnums=[0,1], has_aux=True)(cur_Q, cur_Phi)
        H_grads = jnp.array(H_grads)

        def decoder_postprocessor(graph: jraph.GraphsTuple, aux_data):
            H, cur_state = aux_data
            # integrator = LCIntegrator(self.dt, self.num_mp_steps, self.norm_stats, self.integration_method)

            def dynamics_function(state, t):
                """ 
                    TODO: move to LCIntegrator.dynamics_function?
                """
                state = jnp.array(state)
                dH, _ = jax.grad(H_from_state, argnums=[0,1], has_aux=True)(state[0], state[1])
                dH = jnp.array(dH).reshape(-1,1)
                J = jnp.array([[0, 1],
                               [-1, 0]])
                return jnp.matmul(J, dH).squeeze()
            
            if self.integration_method == 'rk4':
                next_state = rk4(dynamics_function, cur_state, 0.0, self.dt)
            elif self.integration_method == 'euler':
                next_state = euler(dynamics_function, cur_state, 0.0, self.dt)
            else:
                raise NotImplementedError()
            
            # next_state = integrator.dynamics_function(cur_state, H_grad, 0.0)
            # next_V = next_state[0] / self.system_params['C']
            # next_nodes = jnp.array([[0], [next_V]])
            next_edges = jnp.array(next_state).reshape(-1,1)
            next_globals = jnp.array(H)
            graph = graph._replace(edges=next_edges,
                                   globals=next_globals)

            return graph

        aux_data = (H, cur_state)
        processed_graph = decoder_postprocessor(processed_graph, aux_data)
        
        return processed_graph
    
class LC1GNS(nn.Module):
    # Decoder post-processor parameters
    system_params: dict # TODO: CAN REMOVE
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
    noise_std: float = 0.0003

    @nn.compact
    def __call__(self, traj_idx, graph, control, rng):
        encoder_node_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers,
                              with_layer_norm=self.layer_norm, 
                              activation=self.activation)
        encoder_edgeC_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers,
                               with_layer_norm=self.layer_norm, 
                               activation=self.activation)
        encoder_edgeL_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers,
                               with_layer_norm=self.layer_norm, 
                               activation=self.activation)
        encoder_edgeI_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers,
                               with_layer_norm=self.layer_norm, 
                               activation=self.activation)

        decoder_node_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers + [self.node_output_size],
                              activation=self.activation)
        decoder_edgeC_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers + [self.edge_output_size], 
                              activation=self.activation)
        decoder_edgeL_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers + [self.edge_output_size], 
                              activation=self.activation)
        decoder_edgeI_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers + [self.edge_output_size], 
                               activation=self.activation)

        net = CustomEdgeGraphNetworkSimulator(
            edge_idxs=np.array([[0,2], [1,None]]), # TODO: lengths need to be the same...
            encoder_node_fn=encoder_node_fn,
            encoder_edge_fns=[encoder_edgeC_fn, encoder_edgeL_fn, encoder_edgeI_fn],
            decoder_node_fn=decoder_node_fn,
            decoder_edge_fns=[decoder_edgeC_fn, decoder_edgeL_fn, decoder_edgeI_fn],
            decoder_postprocessor=lambda x, _: x, # identity decoder postprocessor
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

        num_edges = len(graph.edges)
        cur_nodes = graph.nodes
        cur_state = graph.edges[:,0]
        cur_Q1, cur_Phi1, cur_Q3, cur_control = cur_state
        if self.training: 
            rng, state_rng, voltage_rng = jax.random.split(rng, 3)
            state_noise = self.noise_std * jax.random.normal(state_rng, (num_edges,))
            noisy_state = cur_state + state_noise

            new_edges = jnp.array(noisy_state).reshape(-1, 1)
            graph = graph._replace(edges=new_edges)

        def H_from_state(Q1, Phi1):
            '''
                1. state to graph
                2. processed_graph = net(graph, aux_data, rng)
                3. return Hamiltonian (processed_graph.globals?)
            '''
            n_node = jnp.array([3])
            n_edge = jnp.array([4])
            senders = jnp.array([0, 1, 0, 2])
            receivers = jnp.array([1, 2, 2, 2])

            # From LCGraphBuilder
            Q3 = -Q1 # TODO: make into independent state (learn that Q3 = -Q1)
            edges = jnp.array([[Q1], [Phi1], [Q3], [cur_control]]) # TODO: control.shape == ()
            global_context = None

            graph = jraph.GraphsTuple(
                nodes=cur_nodes,
                edges=edges,
                globals=global_context,
                n_node=n_node,
                n_edge=n_edge,
                senders=senders,
                receivers=receivers,
            )
            aux_data = None
            processed_graph = net(graph, aux_data, rng)
            return jnp.sum(processed_graph.edges[:-1]), processed_graph # exclude self loops
        
        H, _ = H_from_state(cur_Q1, cur_Phi1)
        H_grads, processed_graph = jax.grad(H_from_state, argnums=[0,1], has_aux=True)(cur_Q1, cur_Phi1)
        H_grads = jnp.array(H_grads)

        def decoder_postprocessor(graph: jraph.GraphsTuple, aux_data):
            H, cur_state = aux_data
            cur_control = cur_state[-1]
            def dynamics_function(state, t):
                """ 
                    TODO: move to LCIntegrator.dynamics_function?
                """
                state = jnp.array(state)
                dH, _ = jax.grad(H_from_state, argnums=[0,1], has_aux=True)(state[0], state[1])
                dH = jnp.array(dH).reshape(-1,1)
                J = jnp.array([[0, 1],
                               [-1, 0]])
                g = jnp.array([[1, 0], [0, 0]])
                return jnp.matmul(J, dH).squeeze() + jnp.matmul(g, jnp.array([cur_control, 0]))
            
            if self.integration_method == 'rk4':
                # TODO: RK4 - fix time index? get state(t)
                state = jnp.array([cur_state[0], cur_state[1]])
                next_state = rk4(dynamics_function, cur_state, 0.0, self.dt) # TODO: can add t as input
            elif self.integration_method == 'euler':
                state = jnp.array([cur_state[0], cur_state[1]])
                next_state = euler(dynamics_function, state, 0.0, self.dt)
            else:
                raise NotImplementedError()
            
            next_Q1 = next_state[0]
            next_Phi1 = next_state[1]
            next_Q3 = -next_Q1 # TODO: learn that Q3 = -Q1
            next_edges = jnp.array([next_Q1, next_Phi1, next_Q3, control]).reshape(-1,1)
            next_globals = jnp.array(H)
            graph = graph._replace(edges=next_edges,
                                   globals=next_globals)

            return graph

        aux_data = (H, cur_state)
        processed_graph = decoder_postprocessor(processed_graph, aux_data)
        
        return processed_graph
    
class LC2GNS(nn.Module):
    # Decoder post-processor parameters
    system_params: dict
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
    noise_std: float = 0.0003

    @nn.compact    
    def __call__(self, traj_idx, graph, control, rng):
        encoder_node_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers,
                              with_layer_norm=self.layer_norm, 
                              activation=self.activation)
        encoder_edgeC_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers,
                               with_layer_norm=self.layer_norm, 
                               activation=self.activation)
        encoder_edgeL_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers,
                               with_layer_norm=self.layer_norm, 
                               activation=self.activation)
        encoder_edgeI_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers,
                               with_layer_norm=self.layer_norm, 
                               activation=self.activation)

        decoder_node_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers + [self.node_output_size],
                              activation=self.activation)
        decoder_edgeC_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers + [self.edge_output_size], 
                              activation=self.activation)
        decoder_edgeL_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers + [self.edge_output_size], 
                              activation=self.activation)
        decoder_edgeI_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers + [self.edge_output_size], 
                              activation=self.activation)

        net = CustomEdgeGraphNetworkSimulator(
            edge_idxs=np.array([[2],[1]]),
            encoder_node_fn=encoder_node_fn,
            encoder_edge_fns=[encoder_edgeC_fn, encoder_edgeL_fn, encoder_edgeI_fn],
            decoder_node_fn=decoder_node_fn,
            decoder_edge_fns=[decoder_edgeC_fn, decoder_edgeL_fn, decoder_edgeI_fn],
            decoder_postprocessor=lambda x, _: x, # identity decoder postprocessor
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

        num_edges = len(graph.edges)
        next_control = control
        cur_nodes = graph.nodes
        cur_Volt = graph.nodes[0].squeeze()
        cur_state = graph.edges[:,0]
        cur_I, cur_Phi2, cur_Q2, cur_I_ext = cur_state
        cur_control = jnp.array([cur_I_ext, cur_Volt])
        if self.training: 
            rng, state_rng, control_rng = jax.random.split(rng, 3)
            state_noise = self.noise_std * jax.random.normal(state_rng, (num_edges,))
            noisy_state = cur_state + state_noise
            new_edges = jnp.array(noisy_state).reshape(-1, 1)
            graph = graph._replace(edges=new_edges)
            control_noise = self.noise_std * jax.random.normal(state_rng)
            control = control + control_noise
        
        def H_from_state(Q2, Phi2):
            '''
                1. state to graph
                2. processed_graph = net(graph, aux_data, rng)
                3. return Hamiltonian (processed_graph.globals?)
            '''
            n_node = jnp.array([3])
            n_edge = jnp.array([num_edges])
            senders = jnp.array([2, 1, 2, 1])
            receivers = jnp.array([0, 0, 1, 1])

            # From LCGraphBuilder
            edges = jnp.array([[cur_I], [Phi2], [Q2], [cur_I_ext]])
            global_context = None

            graph = jraph.GraphsTuple(
                nodes=cur_nodes,
                edges=edges,
                globals=global_context,
                n_node=n_node,
                n_edge=n_edge,
                senders=senders,
                receivers=receivers,
            )
            aux_data = None
            processed_graph = net(graph, aux_data, rng)
            H = jnp.sum(processed_graph.edges[jnp.array([1,2])])
            return H, processed_graph
        
        H, processed_graph = H_from_state(cur_Q2, cur_Phi2)

        def decoder_postprocessor(graph: jraph.GraphsTuple, aux_data):
            H, cur_state = aux_data
            def dynamics_function(state, t):
                """ 
                    TODO: move to LCIntegrator.dynamics_function?
                """
                state = jnp.array(state)
                dH, _ = jax.grad(H_from_state, argnums=[0,1], has_aux=True)(state[0], state[1])
                dH = jnp.array(dH).reshape(-1,1)
                J = jnp.array([[0, 1],
                               [-1, 0]])
                g = jnp.array([[1,0],
                               [0,-1]])
                return jnp.matmul(J, dH).squeeze() + jnp.matmul(g, cur_control).squeeze()
            
            if self.integration_method == 'rk4':
                # TODO: cur_state has dim 3, dynamics_function returns dim 2
                next_state = rk4(dynamics_function, cur_state, 0.0, self.dt)
            elif self.integration_method == 'euler':
                state = jnp.array([cur_state[2], cur_state[1]]) # same sequence as edges
                next_state = euler(dynamics_function, state, 0.0, self.dt)
            else:
                raise NotImplementedError()
            
            next_Q2 = next_state[0]
            next_Phi2 = next_state[1]
            next_I = next_Phi2 / self.system_params['L'][traj_idx].squeeze() # TODO: learn this relation!
            next_I_ext, next_Volt = next_control 
            next_edges = jnp.array([next_I, next_Phi2, next_Q2, next_I_ext]).reshape(-1,1)
            nodes = graph.nodes
            next_nodes = jnp.concatenate((jnp.array([[next_Volt]]), graph.nodes[1:]), axis=0)
            next_globals = jnp.array(H)
            graph = graph._replace(edges=next_edges,
                                   nodes=next_nodes,
                                   globals=next_globals)

            return graph

        aux_data = (H, cur_state)
        processed_graph = decoder_postprocessor(processed_graph, aux_data)
        
        return processed_graph

class CoupledLCGNS(nn.Module):
    """
        Graph Network Simulator (with separate decoder for different edges) for the coupled LC circuit
    """
    # Decoder post-processor parameters
    system_params: dict
    integration_method: str
    dt: float = 0.01

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
    noise_std: float = 0.0003        
    
    @nn.compact
    def __call__(self, traj_idx, graph, control, rng):
        encoder_node_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers,
                              with_layer_norm=self.layer_norm, 
                              activation=self.activation)
        encoder_edge_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers,
                              with_layer_norm=self.layer_norm, 
                              activation=self.activation)

        decoder_node_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers + [self.node_output_size],
                              activation=self.activation)
        decoder_edgeC_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers + [self.edge_output_size], 
                              activation=self.activation)
        decoder_edgeL_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers + [self.edge_output_size], 
                              activation=self.activation)

        net = CustomEdgeGraphNetworkSimulator(
            edge_idxs=[0,2,4],
            encoder_node_fn=encoder_node_fn,
            encoder_edge_fn=encoder_edge_fn,
            decoder_node_fn=decoder_node_fn,
            decoder_edge_fn_1=decoder_edgeC_fn,
            decoder_edge_fn_2=decoder_edgeL_fn,
            decoder_postprocessor=lambda x, _: x, # identity decoder postprocessor
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

        num_edges = len(graph.edges)
        cur_v = graph.nodes[:,0]
        cur_state = graph.edges[:,0]
        cur_Q1, cur_Phi1, cur_Q3, cur_Phi2, cur_Q2 = cur_state
        if self.training: 
            rng, state_rng, voltage_rng = jax.random.split(rng, 3)
            state_noise = self.noise_std * jax.random.normal(state_rng, (num_edges,))
            noisy_state = cur_state + state_noise
            # TODO: add noise to voltage (nodes?)

            new_edges = jnp.array(noisy_state).reshape(-1, 1)
            graph = graph._replace(edges=new_edges)

        def H_from_state(Q1, Phi1, Q2, Phi2):
            '''
                1. state to graph
                2. processed_graph = net(graph, aux_data, rng)
                3. return Hamiltonian (processed_graph.globals?)
            '''
            n_node = jnp.array([4])
            n_edge = jnp.array([5])
            senders = jnp.array([0, 1, 0, 3, 0])
            receivers = jnp.array([1, 2, 2, 2, 3])

            # From LCGraphBuilder
            # Q3 = -(Q1 + Q2) # TODO: explicitly add KCL?
            V2 = Q1 / self.system_params['C'][traj_idx].squeeze()
            V4 = Q2 / self.system_params['C'][traj_idx].squeeze()
            Q3 = -(Q1 + Q2)
            V3 = Q3 / self.system_params['C_prime'][traj_idx].squeeze()

            nodes = jnp.array([[0], [V2], [V3], [V4]])
            edges = jnp.array([[Q1], [Phi1], [Q3], [Phi2], [Q2]])
            global_context = None

            graph = jraph.GraphsTuple(
                nodes=nodes,
                edges=edges,
                globals=global_context,
                n_node=n_node,
                n_edge=n_edge,
                senders=senders,
                receivers=receivers,
            )
            aux_data = None
            processed_graph = net(graph, aux_data, rng)
            return jnp.sum(processed_graph.edges), processed_graph
        
        H, _ = H_from_state(cur_Q1, cur_Phi1, cur_Q2, cur_Phi2)
        H_grads, processed_graph = jax.grad(H_from_state, argnums=[0,1,2,3], has_aux=True)(cur_Q1, cur_Phi1, cur_Q2, cur_Phi2)

        def decoder_postprocessor(graph: jraph.GraphsTuple, aux_data):
            H, cur_state = aux_data

            def dynamics_function(state, t):
                """ 
                    TODO: move to CoupledLCIntegrator.dynamics_function?
                    TODO: change here!
                """
                state = jnp.array(state)
                dH, _ = jax.grad(H_from_state, argnums=[0,1,2,3], has_aux=True)(state[0], state[1], state[2], state[3])
                dH = jnp.array(dH).reshape(-1,1)
                J = jnp.array([[0, 1, 0, 0],
                               [-1, 0, 0, 0],
                               [0, 0, 0, 1],
                               [0, 0, -1, 0]])
                return jnp.matmul(J, dH).squeeze()
            
            if self.integration_method == 'rk4':
                # TODO
                next_state = rk4(dynamics_function, cur_state, 0.0, self.dt)
            elif self.integration_method == 'euler':
                state = jnp.array([cur_Q1, cur_Phi1, cur_Q2, cur_Phi2])
                next_state = euler(dynamics_function, state, 0.0, self.dt)
            else:
                raise NotImplementedError()

            next_Q1 = next_state[0]
            next_Phi1 = next_state[1]
            next_Q2 = next_state[2]
            next_Phi2 = next_state[3]
            next_Q3 = -(next_Q1 + next_Q2)
            next_edges = jnp.array([next_Q1, next_Phi1, next_Q3, next_Phi2, next_Q2]).reshape(-1,1)
            next_globals = jnp.array(H)
            graph = graph._replace(edges=next_edges,
                                   globals=next_globals)

            return graph

        aux_data = (H, cur_state)
        processed_graph = decoder_postprocessor(processed_graph, aux_data)
        
        return processed_graph

class CompLCGNS(nn.Module):
    system_params: FrozenConfigDict
    integration_method: str
    dt: float
    state_one: TrainState
    state_two: TrainState 

    @nn.compact
    def __call__(self, traj_idx, graph1, graph2, rng):
        cur_Q1, cur_Phi1, cur_Q3 = graph1.edges.squeeze()
        _, cur_Phi2, cur_Q2 = graph2.edges.squeeze() # TODO: _ is technically Q3 too
                
        def H_from_state(Q1, Phi1, Q2, Phi2):
            # Modify node voltages and edges to satisfy Kirchhoff's laws
            Q3 = -(Q1 + Q2) # Modification: KCL # TODO: add new current!
            V2 = Q1 / self.system_params['C'][traj_idx]
            V3 = Q3 / self.system_params['C_prime'][traj_idx]
            # TODO: satisfy Kirchhoff's laws
            edges1 = jnp.array([[Q1], [Phi1], [Q3]])
            nodes1 = jnp.array([[0], [V2], [V3]])
            globals1 = None
            senders1 = jnp.array([0, 1, 0])
            receivers1 = jnp.array([1, 2, 2])
            n_node1 = jnp.array([len(nodes1)])
            n_edge1 = jnp.array([len(edges1)])

            V4 = Q2 / self.system_params['C']

            edges2 = jnp.array([[Q2], [Phi2], [Q2]]) # TODO: Q2 or Q3 - first try Q2
            nodes2 = jnp.array([[0], [V4], [0]]) # Modification: same voltage at merged nodes
            globals2 = None
            senders2 = jnp.array([2, 1, 2])
            receivers2 = jnp.array([0, 0, 1])
            n_node2 = jnp.array([len(nodes2)])
            n_edge2 = jnp.array([len(edges2)])

            graph1 = jraph.GraphsTuple(nodes=nodes1,
                                       edges=edges1,
                                       globals=globals1,
                                       senders=senders1, 
                                       receivers=receivers1,
                                       n_node=n_node1,
                                       n_edge=n_edge1)
            
            graph2 = jraph.GraphsTuple(nodes=nodes2,
                                       edges=edges2,
                                       globals=globals2,
                                       senders=senders2, 
                                       receivers=receivers2,
                                       n_node=n_node2,
                                       n_edge=n_edge2)

            # TODO: satisfy KVL (or consistent node voltages) by control!
            control1 = jnp.array(0)
            control2 = jnp.array(V3) # TODO: see if this is correct
            # TODO: SHOULD ONLY APPLY TO GET HAMILTONIAN, NOT GRAPH!
            next_graph1 = self.state_one.apply_fn(self.state_one.params, graph1, control1, rng) 
            next_graph2 = self.state_two.apply_fn(self.state_two.params, graph2, control2, rng)

            H1 = next_graph1.globals.squeeze()
            H2 = next_graph2.globals.squeeze()

            H = H1 + H2

            return H, (next_graph1, next_graph2)
        
        def dynamics_function(state, t):
            dH, _ = jax.grad(H_from_state, argnums=[0,1,2,3], has_aux=True)(state[0], state[1], state[2], state[3])
            dH = jnp.array(dH).reshape(-1,1)
            J = jnp.array([[0, 1, 0, 0],
                           [-1, 0, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, -1, 0]])
            
            return jnp.matmul(J, dH).squeeze()
        
        H, (next_graph1, next_graph2) = H_from_state(cur_Q1, cur_Phi1, cur_Q2, cur_Phi2)
        
        # Integrate port-Hamiltonian dynamics
        next_state = None
        if self.integration_method == 'euler':
            state = jnp.array([cur_Q1, cur_Phi1, cur_Q2, cur_Phi2]) # Q1, Phi1, Q2, Phi2
            next_state = euler(dynamics_function, state, 0, self.dt)

        next_Q1, next_Phi1, next_Q2, next_Phi2 = next_state
        next_Q3 = -(next_Q1 + next_Q2) # KCL
        next_edges1 = jnp.array([[next_Q1], [next_Phi1], [next_Q3]])
        next_graph1 = next_graph1._replace(edges=next_edges1) # nodes will be reset on next forward pass
        next_edges2 = jnp.array([[next_Q2],[next_Phi2], [next_Q2]]) # TODO: first try Q2 at next_edges2[0]
        next_graph2 = next_graph2._replace(edges=next_edges2) # nodes will be reset on next forward pass
        
        return next_graph1, next_graph2
        
class OldGraphNetworkSimulator(nn.Module):
    """ 
        EncodeProcessDecode GN 
    """
    norm_stats: FrozenConfigDict
    system: FrozenConfigDict

    num_mp_steps: int = 1
    layer_norm: bool = False
    use_edge_model: bool = False
    shared_params: bool = False
    vel_history: int = 5
    control_history: int = 1
    noise_std: float = 0.0003

    globals_output_size: int = 0
    edge_output_size: int = 1
    node_output_size: int = 1
    prediction: str = 'acceleration'
    integration_method: str = 'SemiImplicitEuler'
    
    # MLP parameters
    latent_size: int = 16
    hidden_layers: int = 2
    activation: str = 'relu'
    dropout_rate: float = 0
    training: bool = True

    add_self_loops: bool = False
    add_undirected_edges: bool = False

    dt: float = 0.01 # TODO: set from graphbuilder?

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, control, rng) -> jraph.GraphsTuple:
        # TODO: change next_u to cur_u (control)
        num_nodes = len(graph.nodes)
        position = graph.nodes[:,0]
        control = control[1::2] # get nonzero elements (even indices) corresponding to control input
        if self.training: 
            # Add noise to current position (first node feature)
            rng, pos_rng, u_rng = jax.random.split(rng, 3)
            pos_noise = self.noise_std * jax.random.normal(pos_rng, (num_nodes,))
            position = position + pos_noise
            # Add noise to control input at current time-step (next_u)
            # control_noise = self.noise_std * jax.random.normal(u_rng, (num_nodes,))
            # control = control + control_noise

        new_nodes = jnp.column_stack((position, graph.nodes[:,1:], control))
        graph = graph._replace(nodes=new_nodes)

        if self.system.name == 'MassSpring':
            cur_pos = graph.nodes[:,0]
            cur_vel = graph.nodes[:,self.vel_history]
            prev_vel = graph.nodes[:,1:self.vel_history+1] # includes current velocity
            prev_control = graph.nodes[:,self.vel_history+1:] # includes current u

        def update_node_fn(nodes, senders, receivers, globals_):
            node_feature_sizes = [self.latent_size] * self.hidden_layers
            # input = jnp.concatenate((nodes, senders, receivers, globals_), axis=1)
            input = jnp.concatenate((nodes, senders, receivers), axis=1)
            model = MLP(feature_sizes=node_feature_sizes, 
                        activation=self.activation, 
                        dropout_rate=self.dropout_rate, 
                        deterministic=not self.training,
                        with_layer_norm=self.layer_norm)
            return model(input)

        def update_edge_fn(edges, senders, receivers, globals_):
            edge_feature_sizes = [self.latent_size] * self.hidden_layers
            # input = jnp.concatenate((edges, senders, receivers, globals_), axis=1)
            input = jnp.concatenate((edges, senders, receivers), axis=1)
            model = MLP(feature_sizes=edge_feature_sizes,
                        activation=self.activation,
                        dropout_rate=self.dropout_rate, 
                        deterministic=not self.training,
                        with_layer_norm=self.layer_norm)
            return model(input)
            
        def update_global_fn(nodes, edges, globals_):
            del nodes, edges
            time = globals_[0]
            static_params = globals_[1:]
            globals_ = jnp.concatenate((jnp.array([time + self.num_mp_steps]), static_params))
            return globals_ 
        
        # Encoder
        encoder = jraph.GraphMapFeatures(
            embed_edge_fn=MLP(feature_sizes=[self.latent_size] * self.hidden_layers,
                              with_layer_norm=self.layer_norm, 
                              activation=self.activation),
            embed_node_fn=MLP(feature_sizes=[self.latent_size] * self.hidden_layers,
                              with_layer_norm=self.layer_norm, 
                              activation=self.activation),
        )
        
        # Processor
        if not self.use_edge_model:
            update_edge_fn = None

        if graph.globals is None:
            update_global_fn = None

        num_nets = self.num_mp_steps if not self.shared_params else 1
        processor_nets = []
        for _ in range(num_nets): # TODO replace with scan
            net = jraph.GraphNetwork(
                update_node_fn=update_node_fn,
                update_edge_fn=update_edge_fn,
                update_global_fn=update_global_fn,
            )
            processor_nets.append(net)

        # Decoder
        # TODO: custom GraphMapFeatures to differentiate between different edges (e.g. capacitor vs inductor)?
        decoder = jraph.GraphMapFeatures(
            embed_node_fn=MLP(
                feature_sizes=[self.latent_size] * self.hidden_layers + [self.node_output_size],
                activation=self.activation),
            embed_edge_fn=MLP(
                feature_sizes=[self.latent_size] * self.hidden_layers + [self.edge_output_size], 
                activation=self.activation),
        )

        def decoder_postprocessor(graph: jraph.GraphsTuple):
            next_nodes = None
            next_edges = None

            if self.system.name == 'MassSpring':
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

        # Encode features to latent space
        processed_graph = encoder(graph)
        prev_graph = processed_graph
        # Message passing
        for i in range(num_nets): 
            processed_graph = processor_nets[i](processed_graph)
            processed_graph = processed_graph._replace(nodes=processed_graph.nodes + prev_graph.nodes,
                                                       edges=processed_graph.edges + prev_graph.edges)
            prev_graph = processed_graph

        # Decode latent space features back to node features
        processed_graph = decoder(processed_graph)

        # Decoder post-processor
        processed_graph = decoder_postprocessor(processed_graph)

        return processed_graph