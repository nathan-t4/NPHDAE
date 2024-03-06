import jraph
import jax
import diffrax
import flax
import flax.linen as nn

import numpy as np

import jax.numpy as jnp

from typing import Sequence
from ml_collections import FrozenConfigDict

from utils.graph_utils import add_edges

class MLP(nn.Module):
    feature_sizes: Sequence[int]
    activation: str = 'swish'
    dropout_rate: float = 0
    deterministic: bool = True
    with_layer_norm: bool = False

    @nn.compact
    def __call__(self, inputs, training: bool=False):
        x = inputs
        if self.activation == 'swish':
            activation_fn = nn.swish
        elif self.activation == 'relu':
            activation_fn = nn.relu
        else:
            activation_fn = nn.softplus

        for i, size in enumerate(self.feature_sizes):
            x = nn.Dense(features=size)(x)
            if i != len(self.feature_sizes) - 1:
                x = activation_fn(x)
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        if self.with_layer_norm:
            x = nn.LayerNorm()(x)
        return x
    
class NeuralODE(nn.Module):
    ''' 
        Simple Neural ODE
         - https://github.com/patrick-kidger/diffrax/issues/115
         - https://github.com/google/flax/discussions/2891
    '''
    derivative_net: MLP
    solver: diffrax._solver = diffrax.Tsit5()

    @nn.compact
    def __call__(self, inputs, ts=[0,1]):
        if self.is_initializing():
            self.derivative_net(jnp.concatenate((inputs, jnp.array([0]))))
        derivative_net_params = self.derivative_net.variables

        def derivative_fn(t, y, params):
            input = jnp.concatenate((y, jnp.array(t).reshape(-1)))
            return self.derivative_net.apply(params, input)
        
        term = diffrax.ODETerm(derivative_fn)

        solution = diffrax.diffeqsolve(
            term,
            self.solver, 
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1]-ts[0],
            y0=inputs,
            args=derivative_net_params,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts))
                
        return solution.ys


class GraphNet(nn.Module):
    """ 
        EncodeProcessDecode GN 
        TODO: add noise
    """
    normalization_stats: FrozenConfigDict

    num_mp_steps: int = 1
    layer_norm: bool = False
    use_edge_model: bool = False
    shared_params: bool = False

    globals_output_size: int = 0
    edge_output_size: int = 1
    node_output_size: int = 1
    prediction: str = 'acceleration'
    integration_method: str = 'semi_implicit_euler'
    
    # MLP parameters
    latent_size: int = 16
    hidden_layers: int = 2
    activation: str = 'softplus'
    dropout_rate: float = 0
    training: bool = True

    add_self_loops: bool = False
    add_undirected_edges: bool = False

    dt: float = num_mp_steps * 0.01

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        cur_pos = graph.nodes[:,0].reshape(-1)
        if self.prediction == 'acceleration':
            cur_vel = graph.nodes[:,-1].reshape(-1)
            prev_vel = graph.nodes[:,1:]
        elif self.prediction == 'position':
            prev_pos = graph.nodes[:,1:]

        def update_node_fn(nodes, senders, receivers, globals_):
            node_feature_sizes = [self.latent_size] * self.hidden_layers
            inputs = jnp.concatenate((nodes, senders, receivers, globals_), axis=1)
            model = MLP(feature_sizes=node_feature_sizes, 
                        activation=self.activation, 
                        dropout_rate=self.dropout_rate, 
                        deterministic=not self.training,
                        with_layer_norm=self.layer_norm)
            return model(inputs)

        def update_edge_fn(edges, senders, receivers, globals_):
            edge_feature_sizes = [self.latent_size] * self.hidden_layers
            inputs = jnp.concatenate((edges, senders, receivers, globals_), axis=1)
            model = MLP(feature_sizes=edge_feature_sizes,
                        activation=self.activation,
                        dropout_rate=self.dropout_rate, 
                        deterministic=not self.training,
                        with_layer_norm=self.layer_norm)
            return model(inputs)
            
        def update_global_fn(nodes, edges, globals_):
            del nodes, edges
            time = globals_[0]
            static_params = globals_[1:]
            globals_ = jnp.concatenate((jnp.array([time + 1]), static_params))
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

        num_nets = self.num_mp_steps if not self.shared_params else 1
        processor_nets = []
        for _ in range(num_nets): # TODO replace with scan
            net = jraph.GraphNetwork(
                update_node_fn=update_node_fn,
                update_edge_fn=update_edge_fn,
                update_global_fn=update_global_fn,
            )
            processor_nets.append(net)

        # Decoder # TODO: maybe remove hidden layers?
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
            if self.prediction == 'acceleration':
                # Use predicted acceleration to update node features using semi-implicit Euler integration
                if self.integration_method == 'semi_implicit_euler':
                    if self.layer_norm:
                        normalized_acc = graph.nodes.reshape(-1)
                        pred_acc = normalized_acc * self.normalization_stats.acceleration.std + self.normalization_stats.acceleration.mean
                    else:
                        pred_acc = graph.nodes.reshape(-1)
                    next_vel = cur_vel + pred_acc * self.dt
                    next_pos = cur_pos + next_vel * self.dt
                    next_nodes = jnp.column_stack((next_pos, prev_vel[:,1:], next_vel, pred_acc))
                    next_edges = jnp.diff(next_pos).reshape(-1,1)
                    
                elif self.integration_method == 'verlet':
                    # TODO: test - need to change node features for this
                    normalized_acc = graph.nodes.reshape(-1)
                    next_pos = 2 * cur_pos - prev_pos + pred_acc * self.dt**2
                    pred_acc = normalized_acc * self.normalization_stats.acceleration.std + self.normalization_stats.acceleration.mean
                    next_nodes = jnp.column_stack([next_pos, cur_pos, pred_acc])
                    next_edges = jnp.diff(next_pos).reshape(-1,1)

                else:
                    raise RuntimeError('Invalid acceleration decoder postprocessor')
            
            elif self.prediction == 'position':
                if self.layer_norm:
                    normalized_pos = graph.nodes.reshape(-1)
                    pred_pos = normalized_pos * self.normalization_stats.position.std + self.normalization_stats.position.mean
                else:
                    pred_pos = graph.nodes.reshape(-1)

                next_vel = (pred_pos - cur_pos) / self.dt

                next_nodes = jnp.column_stack((pred_pos, prev_pos[:,1:], cur_pos))
                next_edges = jnp.diff(pred_pos).reshape(-1, 1)
            
            if self.add_undirected_edges:
                next_edges = jnp.concatenate((next_edges, next_edges), axis=0)
            
            if self.add_self_loops:
                next_edges = jnp.concatenate((next_edges, jnp.zeros((3, 1))), axis=0)

            graph = graph._replace(nodes=next_nodes,
                                   edges=next_edges)            
            return graph

        # Encode features to latent space
        processed_graph = encoder(graph)

        # Message passing
        for i in range(self.num_mp_steps): 
            processed_graph = processor_nets[i](processed_graph)

        # def mp_step(carry, net):
        #     return net(carry), _
        # processed_graph, _ = jax.lax.scan(mp_step, processed_graph, processor_nets)
        # Decode latent space features back to node features
        processed_graph = decoder(processed_graph)

        # Decoder post-processor
        processed_graph = decoder_postprocessor(processed_graph)

        return processed_graph
    
class GNODE(nn.Module):
    """ 
        EncodeProcessDecode GNODE

        Graph Neural Network with Neural ODE message passing functions
        
        The neural ODEs takes concatenated latent features as input and its output is passed through a linear layer

        TODO:
        - fix neural ODE call - time explicitly given (from globals)?
        
    """
    normalization_stats: FrozenConfigDict

    num_mp_steps: int = 1
    layer_norm: bool = False
    use_edge_model: bool = False
    shared_params: bool = False

    globals_output_size: int = 0
    edge_output_size: int = 1
    node_output_size: int = 1
    
    # MLP parameters
    latent_size: int = 16
    hidden_layers: int = 2
    dropout_rate: float = 0
    training: bool = True

    add_self_loops: bool = False
    add_undirected_edges: bool = False

    dt: float = num_mp_steps * 0.01

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        cur_pos = graph.nodes[:,0].reshape(-1)
        cur_vel = graph.nodes[:,-1].reshape(-1)
        prev_vel = graph.nodes[:,1:]
        def update_node_fn(nodes, senders, receivers, globals_):
            inputs = jnp.concatenate((nodes, senders, receivers, globals_), axis=1)
            node_feature_sizes = [inputs.shape[1]] * self.hidden_layers
            derivative_net = MLP(feature_sizes=node_feature_sizes, 
                                 activation='relu', 
                                 dropout_rate=self.dropout_rate, 
                                 deterministic=not self.training)
            model = NeuralODE(derivative_net)
            output = model(inputs).squeeze()
            return nn.Dense(self.latent_size)(output)

        def update_edge_fn(edges, senders, receivers, globals_):
            inputs = jnp.concatenate((edges, senders, receivers, globals_), axis=1)
            edge_feature_sizes = [inputs.shape[1]] * self.hidden_layers
            derivative_net = MLP(feature_sizes=edge_feature_sizes, 
                                 activation='relu',
                                 dropout_rate=self.dropout_rate, 
                                 deterministic=not self.training)
            model = NeuralODE(derivative_net)
            output = model(inputs).squeeze()
            return nn.Dense(self.latent_size)(output)
            
        def update_global_fn(nodes, edges, globals_):
            del nodes, edges
            time = globals_[0]
            static_params = globals_[1:]
            globals_ = jnp.concatenate((jnp.array([time + 1]), static_params))
            return globals_ 
        
        if not self.use_edge_model:
            update_edge_fn = None

       # Encoder
        encoder = jraph.GraphMapFeatures(
            embed_edge_fn=MLP(feature_sizes=[self.latent_size] * self.hidden_layers,
                              with_layer_norm=self.layer_norm),
            embed_node_fn=MLP(feature_sizes=[self.latent_size] * self.hidden_layers,
                              with_layer_norm=self.layer_norm),
        )
        
        # Processor
        if not self.use_edge_model:
            update_edge_fn = None

        num_nets = self.num_mp_steps if not self.shared_params else 1
        processor_nets = []
        for _ in range(num_nets):
            net = jraph.GraphNetwork(
                update_node_fn=update_node_fn,
                update_edge_fn=update_edge_fn,
                update_global_fn=update_global_fn,
            )
            processor_nets.append(net)

        # Decoder
        decoder = jraph.GraphMapFeatures(
            embed_node_fn=MLP(feature_sizes=[self.latent_size] * self.hidden_layers + [self.node_output_size]),
            embed_edge_fn=MLP(feature_sizes=[self.latent_size] * self.hidden_layers + [self.edge_output_size]),
        )

        def decoder_postprocessor(graph: jraph.GraphsTuple, mode='semi_implicit_euler'):
            # Use predicted acceleration to update node features using semi-implicit Euler integration
            next_nodes = None
            next_edges = None
            if mode == 'semi_implicit_euler':
                if self.layer_norm:
                    normalized_acc = graph.nodes.reshape(-1)
                    pred_acc = normalized_acc * self.normalization_stats.acceleration.std + self.normalization_stats.acceleration.mean
                else:
                    pred_acc = graph.nodes.reshape(-1)
                next_vel = cur_vel + pred_acc * self.dt
                next_pos = cur_pos + next_vel * self.dt

                next_nodes = jnp.column_stack([next_pos, prev_vel[:,1:], next_vel, pred_acc])
                next_edges = jnp.diff(next_pos).reshape(-1,1)
                
            elif mode == 'verlet':
                # TODO: test - need to change node features for this
                pred_acc = graph.nodes.reshape(-1)
                next_pos = 2 * cur_pos - prev_pos + pred_acc * self.dt**2

                # TODO: normalize acceleration and put it into next_node (to normalize reward fun)
                normalized_acc = pred_acc # find mean and std of pred_acc from graphs

                next_nodes = jnp.column_stack([next_pos, cur_pos, pred_acc])
                next_edges = jnp.diff(next_pos).reshape(-1,1)

            else:
                raise RuntimeError('Invalid decoder postprocessor')
            
            if self.add_undirected_edges:
                next_edges = jnp.concatenate((next_edges, next_edges), axis=0)
            
            if self.add_self_loops:
                next_edges = jnp.concatenate((next_edges, jnp.zeros((3, 1))), axis=0)

            graph = graph._replace(nodes=next_nodes,
                                   edges=next_edges)           
            return graph

        # Encode features to latent space
        processed_graph = encoder(graph)

        # Message passing
        for i in range(self.num_mp_steps):
            processed_graph = processor_nets[i](processed_graph)

        # Decode latent space features back to node features
        processed_graph = decoder(processed_graph)

        # Decoder post-processor
        processed_graph = decoder_postprocessor(processed_graph)

        return processed_graph