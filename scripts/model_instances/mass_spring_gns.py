import jax
import jraph
import jax.numpy as jnp
import flax.linen as nn
from ml_collections import FrozenConfigDict
from scripts.models.mlp import *
from scripts.models.gnn import GraphNetworkSimulator
from utils.models_utils import MassSpringIntegrator

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