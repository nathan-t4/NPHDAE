import jraph
import diffrax
import flax.linen as nn

import numpy as np

import jax.numpy as jnp

from typing import Sequence

class MLP(nn.Module):
    feature_sizes: Sequence[int]
    activation: str = 'swish'
    dropout_rate: float = 0
    deterministic: bool = True

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        if self.activation == 'swish':
            activation_fn = nn.swish
        else:
            activation_fn = nn.relu

        for i, size in enumerate(self.feature_sizes):
            x = nn.Dense(features=size)(x)
            if i != len(self.feature_sizes) - 1:
                x = activation_fn(x)
                x = nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic)(x)
        return x
    
class NeuralODE(nn.Module):
    ''' 
        Simple Neural ODE
         - https://github.com/patrick-kidger/diffrax/issues/115
         - https://github.com/google/flax/discussions/2891
    '''
    derivative_net: MLP
    solver: diffrax._solver = diffrax.Dopri8()

    @nn.compact
    def __call__(self, inputs):
        if self.is_initializing():
            self.derivative_net(inputs)
        
        derivative_net_params = self.derivative_net.variables["params"]

        def derivative_fn(t, y, params):
            return self.derivative_net.apply({'params': params}, y)
        
        term = diffrax.ODETerm(derivative_fn)

        solution = diffrax.diffeqsolve(
            term,
            self.solver, 
            t0=0,
            t1=1,
            dt0=0.1,
            y0=inputs,
            args=derivative_net_params)
        
        return solution.ys
    

class GraphNet(nn.Module):
    """ EncodeProcessDecode GN """
    num_message_passing_steps: int = 1
    layer_norm: bool = False
    use_edge_model: bool = False
    use_global_model: bool = False

    globals_output_size: int = 0
    edge_output_size: int = 1
    node_output_size: int = 1
    # MLP parameters
    latent_size: int = 10
    hidden_layers: int = 2
    dropout_rate: float = 0
    deterministic: bool = True

    dt: float = 1.0

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        cur_pos = graph.nodes[:,0]
        cur_vel = graph.nodes[:,1]

        def update_node_fn(nodes, senders, receivers, globals_):
            node_feature_sizes = [self.latent_size] * self.hidden_layers
            if self.use_global_model:
                inputs = jnp.concatenate((nodes, senders, receivers, globals_), axis=1)
            else:
                inputs = jnp.concatenate((nodes, senders, receivers), axis=1)
            # inputs = nodes
            model = MLP(feature_sizes=node_feature_sizes, activation='relu', 
                        dropout_rate=self.dropout_rate, deterministic=self.deterministic)
            return model(inputs)

        def update_edge_fn(edges, senders, receivers, globals_):
            # TODO: difference btw senders and receivers?
            edge_feature_sizes = [self.latent_size] * self.hidden_layers
            if self.use_global_model:
                inputs = jnp.concatenate((edges, senders, globals_), axis=1)
            else:
                inputs = jnp.concatenate((edges, senders), axis=1)
            model = MLP(feature_sizes=edge_feature_sizes, activation='relu',
                        dropout_rate=self.dropout_rate, deterministic=self.deterministic)
            return model(inputs)
            
        def update_global_fn(nodes, edges, globals_):
            del nodes, edges
            # time_idxs = [i % 8 == 0 for i in range(len(globals))]
            # times = [globals_[i] += 1 for i in time_idxs]
            time = globals_[0]
            static_params = globals_[1:]
            globals_ = jnp.concatenate((jnp.array([time]), static_params))
            return globals_ 
        
        if not self.use_edge_model:
            update_edge_fn = None
        
        if not self.use_global_model:
            update_global_fn = None

        encoder = jraph.GraphMapFeatures(
            embed_edge_fn=nn.Dense(features=self.latent_size),
            embed_node_fn=nn.Dense(features=self.latent_size),
        )
        net = jraph.GraphNetwork(
            update_node_fn=update_node_fn,
            update_edge_fn=update_edge_fn,
            update_global_fn=update_global_fn,
        )

        node_decode_fn = nn.Dense(self.node_output_size) if self.node_output_size != 0 else None
        edge_decode_fn = nn.Dense(self.edge_output_size) if self.edge_output_size != 0 else None

        decoder = jraph.GraphMapFeatures(
            embed_node_fn=node_decode_fn,
            embed_edge_fn=edge_decode_fn,
        )

        # Encode features to latent space
        processed_graph = encoder(graph)

        # Message passing
        for _ in range(self.num_message_passing_steps):
            processed_graph = net(processed_graph)
        
        # Layer normalization
        if self.layer_norm:
            # no layer normalization for globals since it is time + static params
            processed_graph = processed_graph._replace(
                nodes=nn.LayerNorm()(processed_graph.nodes),
                edges=nn.LayerNorm()(processed_graph.edges), 
                # globals=nn.LayerNorm()(processed_graph.globals)
            )
        
        # Decode latent space features back to node features
        processed_graph = decoder(processed_graph)

        def decoder_postprocessor(graph: jraph.GraphsTuple):
            # Use predicted acceleration to update node features using Euler integration
            pred_acc = graph.nodes.reshape(-1)
            next_vel = cur_vel + pred_acc * self.dt
            next_pos = cur_pos + next_vel * self.dt

            graph._replace(nodes=jnp.array([next_pos, next_vel]))

            return graph

        processed_graph = decoder_postprocessor(processed_graph)
        
        return processed_graph
    
class GNODE(nn.Module):
    """ 
        EncodeProcessDecode GNODE

        Graph Neural Network with Neural ODE message passing functions
        
        The neural ODEs takes concatenated latent features as input and its output is passed through a linear layer
    """
    num_message_passing_steps: int = 1
    layer_norm: bool = False
    use_edge_model: bool = False

    globals_output_size: int = 0
    edge_output_size: int = 1
    node_output_size: int = 1
    # MLP parameters
    latent_size: int = 10
    hidden_layers: int = 2
    dropout_rate: float = 0
    deterministic: bool = True

    dt: float = 1.0

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        cur_pos = graph.nodes[:,0]
        cur_vel = graph.nodes[:,1]

        def update_node_fn(nodes, senders, receivers, globals_):
            inputs = jnp.concatenate((nodes, senders, receivers, globals_), axis=1)
            node_feature_sizes = [inputs.shape[1]] * self.hidden_layers
            derivative_net = MLP(feature_sizes=node_feature_sizes, activation='relu', 
                                 dropout_rate=self.dropout_rate, deterministic=self.deterministic)
            model = NeuralODE(derivative_net)
            output = model(inputs).squeeze()
            return nn.Dense(self.latent_size)(output)

        def update_edge_fn(edges, senders, receivers, globals_):
            # TODO: difference btw senders and receivers?
            inputs = jnp.concatenate((edges, senders, globals_), axis=1)
            edge_feature_sizes = [inputs.shape[1]] * self.hidden_layers
            derivative_net = MLP(feature_sizes=edge_feature_sizes, activation='relu',
                                 dropout_rate=self.dropout_rate, deterministic=self.deterministic)
            model = NeuralODE(derivative_net)
            output = model(inputs).squeeze()
            return nn.Dense(self.latent_size)(output)
            
        def update_global_fn(nodes, edges, globals_):
            del nodes, edges
            # time_idxs = [i % 8 == 0 for i in range(len(globals))]
            # times = [globals_[i] += 1 for i in time_idxs]
            time = globals_[0]
            static_params = globals_[1:]
            globals_ = jnp.concatenate((jnp.array([time]), static_params))
            return globals_ 
        
        if not self.use_edge_model:
            update_edge_fn = None

        encoder = jraph.GraphMapFeatures(
            embed_edge_fn=nn.Dense(features=self.latent_size),
            embed_node_fn=nn.Dense(features=self.latent_size),
        )
        net = jraph.GraphNetwork(
            update_node_fn=update_node_fn,
            update_edge_fn=update_edge_fn,
            update_global_fn=update_global_fn,
        )

        node_decode_fn = nn.Dense(self.node_output_size) if self.node_output_size != 0 else None
        edge_decode_fn = nn.Dense(self.edge_output_size) if self.edge_output_size != 0 else None

        decoder = jraph.GraphMapFeatures(
            embed_node_fn=node_decode_fn,
            embed_edge_fn=edge_decode_fn,
        )

        # Encode features to latent space
        processed_graph = encoder(graph)

        # Message passing
        for _ in range(self.num_message_passing_steps):
            processed_graph = net(processed_graph)
        
        # Layer normalization
        if self.layer_norm:
            # no layer normalization for globals since it is time + static params
            processed_graph = processed_graph._replace(
                nodes=nn.LayerNorm()(processed_graph.nodes),
                edges=nn.LayerNorm()(processed_graph.edges), 
                # globals=nn.LayerNorm()(processed_graph.globals)
            )
        
        # Decode latent space features back to node features
        processed_graph = decoder(processed_graph)

        def decoder_postprocessor(graph: jraph.GraphsTuple):
            # Use predicted acceleration to update node features using Euler integration
            pred_acc = graph.nodes.reshape(-1)
            next_vel = cur_vel + pred_acc * self.dt
            next_pos = cur_pos + next_vel * self.dt

            graph._replace(nodes=jnp.array([next_pos, next_vel]))

            return graph

        processed_graph = decoder_postprocessor(processed_graph)
        
        return processed_graph