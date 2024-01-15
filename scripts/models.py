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

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        embedder = jraph.GraphMapFeatures(
            embed_edge_fn=nn.Dense(features=5),
            embed_node_fn=nn.Dense(features=5),
            embed_global_fn=nn.Dense(features=5),
        )

        # processed_graphs = embedder(graph)

        def update_edge_fn(edges, senders, receivers, globals_):
            print("Edge shape:", jnp.shape(edges))
            net = MLP(feature_sizes=[jnp.shape(edges),jnp.shape(edges)])
            return net

        def update_node_fn(nodes, senders, receivers, globals_):
            print("Nodes shape:", jnp.shape(nodes))
            # key = jax.random.PRNGKey(seed=0)
            net = MLP(feature_sizes=[jnp.shape(nodes),jnp.shape(nodes)])
            # params = net.init(key)
            # return net.apply(params, nodes)
            return net

        def update_global_fn(nodes, edges, globals_):
            # return globals_
            return None
        
        net = jraph.GraphNetwork(
            update_edge_fn=update_edge_fn,
            update_node_fn=update_node_fn,
            update_global_fn=update_global_fn,
        )

        processed_graphs = net(graph)

        return processed_graphs