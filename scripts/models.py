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
    
    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for size in self.feature_sizes:
            x = nn.Dense(features=size)(x)
            x = nn.relu(x)
        return x

class GraphNet(nn.Module):

    # latent_size: int
    hidden_layers: int = 2

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        nodes, edges, receivers, senders, globals_, n_node, n_edge = graph

        # TODO: use encoder?
        embedder = jraph.GraphMapFeatures(
            embed_edge_fn=nn.Dense(features=5),
            embed_node_fn=nn.Dense(features=5),
            embed_global_fn=nn.Dense(features=5),
        )

        # processed_graphs = embedder(graph)

        edge_feature_sizes = [len(edges)] * self.hidden_layers
        node_features_sizes = [np.shape(nodes)[1]] * self.hidden_layers

        update_edge_fn = jraph.concatenated_args(MLP(feature_sizes=edge_feature_sizes))
        update_node_fn = jraph.concatenated_args(MLP(feature_sizes=node_features_sizes))
        update_global_fn = None

        net = jraph.GraphNetwork(
            update_edge_fn=update_edge_fn,
            update_node_fn=update_node_fn,
            update_global_fn=update_global_fn,
        )

        processed_graphs = net(graph)

        # TODO: add decoder?
        decoder = jraph.GraphMapFeatures(
            embed_node_fn=nn.Dense(np.shape(nodes[1]))
        )

        # processed_graphs = decoder(processed_graphs)


        return processed_graphs