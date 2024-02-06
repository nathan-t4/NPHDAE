import jraph
import jax
import diffrax
import flax
import flax.linen as nn
import equinox as eqx

import numpy as np

from typing import Tuple

from jax.experimental.ode import odeint
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
    
    
class NeuralODE(flax.struct.PyTreeNode):
    ''' 
        Simple Neural ODE - https://github.com/patrick-kidger/diffrax/issues/115
        TODO: work in progress
    '''
    derivative_net: MLP

    def init(self, rng, coords):      
        rng, derivative_net_rng = jax.random.split(rng)
        coords, derivative_net_params = self.derivative_net.init_with_output(derivative_net_rng, coords)

        params = flax.core.FrozenDict({"derivative_net": derivative_net_params})

        return params


    def __call__(self, params, inputs):
        
        def derivative_fn(t, y, args):
            return self.derivative_net.apply(params["derivative_net"], y)
        
        term = diffrax.ODETerm(derivative_fn)

        solution = diffrax.diffeqsolve(
            term,
            diffrax.Euler(), 
            t0=0,
            t1=1,
            dt0=0.1,
            y0=inputs)
        
        return solution.ys
    

class GraphNet(nn.Module):
    """ EncodeProcessDecode GN """
    num_message_passing_steps: int = 1
    layer_norm: bool = False
    use_edge_model: bool = False

    globals_output_size: int = 0
    edge_output_size: int = 1
    node_output_size: int = 1

    latent_size: int = 10
    hidden_layers: int = 2
    dropout_rate: float = 0
    deterministic: bool = True

    dt: float = 1.0

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        def update_node_fn(nodes, senders, receivers, globals_):
            node_feature_sizes = [self.latent_size] * self.hidden_layers
            inputs = jnp.concatenate((nodes, senders, receivers, globals_), axis=1)
            # inputs = nodes
            model = MLP(feature_sizes=node_feature_sizes, activation='relu', 
                        dropout_rate=self.dropout_rate, deterministic=self.deterministic)
            # model = NeuralODE(derivative_net) # TODO
            return model(inputs)

        def update_edge_fn(edges, senders, receivers, globals_):
            # print('senders', senders)
            # # Update edges using Euler integration
            # velocity_magnitudes = np.linalg.norm(senders, axis=0)
            # print('velocity magnitudes', velocity_magnitudes)
            # dq = jnp.diff(velocity_magnitudes) * self.dt
            # # pad with zeros (if necessary)
            # de = jnp.pad(dq, (0, jnp.zeros(len(edges) - len(dq))))
            # new_edges = edges + de
            edge_feature_sizes = [self.latent_size] * self.hidden_layers
            inputs = jnp.concatenate((edges, senders, globals_), axis=1)
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
    

        return processed_graph