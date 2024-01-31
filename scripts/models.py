import jraph
import jax
import flax.linen as nn

import numpy as np

from typing import Tuple

from jax.experimental.ode import odeint
import jax.numpy as jnp

from typing import Sequence

class MLP(nn.Module):
    feature_sizes: Sequence[int]
    activation: str = 'swish'

    @nn.compact
    def __call__(self, inputs):
        x = inputs

        if self.activation == 'swish':
            activation_fn = nn.swish
        else:
            activation_fn = nn.relu

        for size in self.feature_sizes:
            x = nn.Dense(features=size)(x)
            x = activation_fn(x)
        return x

class GraphNet(nn.Module):
    globals_output_size: int = 0
    edge_output_size: int = 1
    node_output_size: int = 1
    latent_size: int = 5
    hidden_layers: int = 2
    layer_norm: bool = True
    dt: float = 1.0
    use_edge_model: bool = False

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # Pass features through a linear layer to latent space
        encoder = jraph.GraphMapFeatures(
            embed_edge_fn=nn.Dense(features=self.latent_size),
            embed_node_fn=nn.Dense(features=self.latent_size),
        )

        processed_graphs = encoder(graph)
        
        node_features_sizes = [self.latent_size] * self.hidden_layers 
        update_node_fn = lambda n, s, r, g: MLP(feature_sizes=node_features_sizes, activation='swish')(n)

        def update_edge_fn(edges, senders, receivers, globals_):
            # Update edges using naive Euler integration
            dq = jnp.array([senders[1] - senders[0], senders[2] - senders[1]], dtype=jnp.float32) * self.dt
            new_edges = dq + edges
            return new_edges
            
        def update_global_fn(nodes, edges, globals_):
            del nodes, edges
            time = globals_[0] + self.dt # increment time
            static_params = globals_[1:]
            globals_ = jnp.concatenate((jnp.array([time]), static_params))
            return globals_ 
        
        if not self.use_edge_model:
            update_edge_fn = None
        
        net = jraph.GraphNetwork(
            update_node_fn=update_node_fn,
            update_edge_fn=update_edge_fn,
            update_global_fn=update_global_fn,
        )

        processed_graphs = net(graph)

        # if self.layer_norm:
        #     # no layer normalization for globals since it is time + static params
        #     processed_graphs = processed_graphs._replace(
        #         nodes=nn.LayerNorm()(processed_graphs.nodes),
        #         edges=nn.LayerNorm()(processed_graphs.edges), 
        #     )

        node_decode_fn = nn.Dense(self.node_output_size) if self.node_output_size != 0 else None
        edge_decode_fn = nn.Dense(self.edge_output_size) if self.edge_output_size != 0 else None

        decoder = jraph.GraphMapFeatures(
            embed_node_fn=node_decode_fn,
            embed_edge_fn=edge_decode_fn,
        )

        processed_graphs = decoder(processed_graphs)

        return processed_graphs