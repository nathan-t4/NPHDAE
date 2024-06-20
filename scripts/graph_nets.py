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
    graph_from_state: Callable
    J: Union[None, Array] # if None then learn
    g: Union[None, Array] # if None then learn

    integration_method: str
    dt: float
    T: int

    # Graph Network parameters
    edge_idxs: Array
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
        encoder_node_fn = vmapMLP(feature_sizes=[self.latent_size] * (self.hidden_layers),
                                  with_layer_norm=self.layer_norm, 
                                  activation=self.activation) # TODO: GraphMapFeatures only batches edges not nodes... 
        encoder_edge1_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers,
                               with_layer_norm=self.layer_norm, 
                               activation=self.activation)
        encoder_edge2_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers,
                               with_layer_norm=self.layer_norm, 
                               activation=self.activation)

        decoder_node_fn = vmapMLP(feature_sizes=[self.latent_size] * self.hidden_layers + [node_output_size],
                                  activation=self.activation)
        decoder_edge1_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers + [edge_output_size], 
                               activation=self.activation)
        decoder_edge2_fn = MLP(feature_sizes=[self.latent_size] * self.hidden_layers + [edge_output_size], 
                               activation=self.activation)

        net = HeterogeneousGraphNetworkSimulator(
            edge_idxs=np.array(self.edge_idxs),
            encoder_node_fn=encoder_node_fn,
            encoder_edge_fns=[encoder_edge1_fn, encoder_edge2_fn],
            decoder_node_fn=decoder_node_fn,
            decoder_edge_fns=[decoder_edge1_fn, decoder_edge2_fn],
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
            layer_norm=self.layer_norm
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

        def decoder_postprocessor(graph: jraph.GraphsTuple, aux_data):
            H, cur_state = aux_data
            def dynamics_function(x, t):

                def get_learned_J():
                    out_dim = len(x)
                    J = nn.Dense(features=out_dim, use_bias=False)(jnp.eye(out_dim))
                    J_triu = jnp.triu(J)
                    return J_triu - J_triu.T  # Make J skew-symmetric
                
                def get_learned_g():
                    out_dim = len(x)
                    g = nn.Dense(features=out_dim, use_bias=False)(jnp.eye(out_dim))
                    return g
                
                dH, _ = jax.grad(H_from_state, has_aux=True)(x)
                z = dH
                J = self.J if self.J is not None else get_learned_J()
                g = self.g if self.g is not None else get_learned_g()
                return jnp.matmul(J, z).squeeze() + jnp.matmul(g, control).squeeze()
            
            if self.T == 1:
                next_state = integrator_factory(self.integration_method)(dynamics_function, cur_state, 0.0, self.dt)
            else: # multi-step integration scheme
                next_state = integrator_factory(self.integration_method)(dynamics_function, cur_state, 0.0, self.dt, self.T)
            
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

class CompLCGNS(nn.Module):
    integration_method: str
    dt: float
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
        if self.integration_method == 'euler':
            next_state = euler(partial(dynamics_function, aux_data=aux_data), full_state, 0, self.dt) 
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