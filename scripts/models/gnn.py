import jraph
import jax
import flax.linen as nn
import jax.numpy as jnp

from typing import Sequence, Callable
from ml_collections import FrozenConfigDict
from utils.graph_utils import *
from utils.jax_utils import *
from utils.models_utils import *

from scripts.models.mlp import *
    
class GraphNetworkSimulator(nn.Module):
    encoder_node_fn: Callable
    encoder_edge_fn: Callable
    decoder_node_fn: Callable
    decoder_edge_fn: Callable
    decoder_postprocessor: Callable

    num_mp_steps: int
    shared_params: bool

    use_edge_model: bool
    
    layer_norm: bool = False
    latent_size: int = 16
    hidden_layers: int = 2
    activation: str = 'relu'
    dropout_rate: float = 0
    training: bool = True

    @nn.compact    
    def __call__(self, graph, aux_data, rng):   
        if aux_data is not None:
            graph = graph._replace(globals=jnp.array([aux_data]))
        
        encoder = jraph.GraphMapFeatures(
            embed_edge_fn=self.encoder_node_fn,
            embed_node_fn=self.encoder_edge_fn,
        )  

        decoder = jraph.GraphMapFeatures(
            embed_node_fn=self.decoder_node_fn,
            embed_edge_fn=self.decoder_edge_fn,
        )   

        def update_node_fn(nodes, senders, receivers, globals_):
            node_feature_sizes = [self.latent_size] * self.hidden_layers
            input = jnp.concatenate((nodes, senders, receivers, globals_), axis=1)
            # input = jnp.concatenate((nodes, senders, receivers), axis=1)
            model = MLP(feature_sizes=node_feature_sizes, 
                        activation=self.activation, 
                        dropout_rate=self.dropout_rate, 
                        deterministic=not self.training,
                        with_layer_norm=self.layer_norm)
            return model(input)

        def update_edge_fn(edges, senders, receivers, globals_):
            edge_feature_sizes = [self.latent_size] * self.hidden_layers
            input = jnp.concatenate((edges, senders, receivers, globals_), axis=1)
            # input = jnp.concatenate((edges, senders, receivers), axis=1)
            model = MLP(feature_sizes=edge_feature_sizes,
                        activation=self.activation,
                        dropout_rate=self.dropout_rate, 
                        deterministic=not self.training,
                        with_layer_norm=self.layer_norm)
            return model(input)
            
        # def update_global_fn(nodes, edges, globals_):
        #     del nodes, edges
        #     time = globals_[0]
        #     static_params = globals_[1:]
        #     globals_ = jnp.concatenate((jnp.array([time + self.num_mp_steps]), static_params))
        #     return globals_ 
        
        if not self.use_edge_model:
            update_edge_fn = None

        # TODO: was graph.globals is None
        update_global_fn = None

        graph_net = jraph.GraphNetwork(
                update_node_fn=update_node_fn,
                update_edge_fn=update_edge_fn,
                update_global_fn=update_global_fn,
            )
        
        num_nets = self.num_mp_steps if not self.shared_params else 1
        graph_nets = []
        for _ in range(num_nets):
            graph_nets.append(graph_net)
        
        # Encode features to latent space
        processed_graph = encoder(graph)
        prev_graph = processed_graph

        # Message passing
        for i in range(num_nets): 
            processed_graph = graph_nets[i](processed_graph)
            processed_graph = processed_graph._replace(nodes=processed_graph.nodes + prev_graph.nodes,
                                                       edges=processed_graph.edges + prev_graph.edges)
            prev_graph = processed_graph

        # Decode latent space features back to node features
        processed_graph = decoder(processed_graph)

        # Decoder post-processor
        processed_graph = self.decoder_postprocessor(processed_graph, aux_data)

        return processed_graph
    
class HeterogeneousGraphNetworkSimulator(nn.Module):
    """
        Graph Network Simulator with two edge decoders. 
        
        Edges with indices on the n-th row of "edge_idxs" 
        are processed with the n-th decoder.
        All other edges are processed with the last decoder.
    """
    encoder_edge_fns: Sequence[Callable]
    encoder_node_fns: Sequence[Callable]
    decoder_edge_fns: Sequence[Callable]
    decoder_node_fns: Sequence[Callable]
    decoder_postprocessor: Callable

    num_mp_steps: int
    shared_params: bool

    learn_nodes: bool
    use_edge_model: bool
    use_global_model: bool
    
    dt: float
    T: int
    layer_norm: bool
    latent_size: int
    hidden_layers: int
    activation: str
    dropout_rate: float
    training: bool

    @nn.compact    
    def __call__(self, graph, aux_data, rng):   
        def HeterogeneousGraphMapFeatures(embed_edge_fns: Sequence[Callable] = None,
                                          embed_node_fns: Sequence[Callable] = None,
                                          embed_global_fn: Callable = None):
            """
                This function processes each edge independently, but the nodes are processed altogether.
                So the edges are effectively batched, while the nodes are not.
                This is why we use vmapMLP for the node encoder/decoder and 
                just MLP for the edge encoders/decoders.
            """
            identity = lambda x : x
            embed_globals_fn = embed_global_fn if embed_global_fn else identity

            def Embed(graph):
                """
                    Differentiate edges and nodes by type, 
                    and apply the correct mapping function for each edge and node feature.
                """
                new_edges = None
                for i in range(len(graph.edges)):
                    # TODO: make embed_edge_fns based on edge indices...
                    new_edge = embed_edge_fns[i](graph.edges[i])
                    if new_edges is None:
                        new_edges = new_edge
                    else:
                        new_edges = jnp.concatenate((new_edges, new_edge))

                new_nodes = None
                for i in range(len(graph.nodes)):
                    new_node = embed_node_fns[0](graph.nodes[i]) # use same embed function for all nodes
                    if new_nodes is None:
                        new_nodes = new_node
                    else:
                        new_nodes = jnp.concatenate((new_nodes, new_node))

                return graph._replace(nodes=new_nodes.reshape(graph.nodes.shape[0], -1),
                                      edges=new_edges.reshape(graph.edges.shape[0], -1),
                                      globals=embed_globals_fn(graph.globals))    
            return Embed
        
        use_global_model = self.use_global_model
        if aux_data is not None:
            graph = graph._replace(globals=jnp.array([aux_data]).reshape(-1,1))
            use_global_model = True
        
        encoder = HeterogeneousGraphMapFeatures(
            embed_node_fns=self.encoder_node_fns,
            embed_edge_fns=self.encoder_edge_fns,
        )  

        decoder = HeterogeneousGraphMapFeatures(
            embed_node_fns=self.decoder_node_fns,
            embed_edge_fns=self.decoder_edge_fns,
        )

        def update_node_fn(nodes, senders, receivers, globals_):
            node_feature_sizes = [self.latent_size] * self.hidden_layers
            if use_global_model:
                input = jnp.concatenate((nodes, senders, receivers, globals_), axis=1)
            elif self.learn_nodes:
                input = jnp.concatenate((senders, receivers), axis=1)
            else:
                input = jnp.concatenate((nodes, senders, receivers), axis=1)
            # vmap across batch dim so that the MLP processed each node separately
            model = vmapMLP(feature_sizes=node_feature_sizes, 
                            activation=self.activation, 
                            dropout_rate=self.dropout_rate, 
                            deterministic=not self.training,
                            with_layer_norm=self.layer_norm)
                            # name='update_node')
            return model(input)

        def update_edge_fn(edges, senders, receivers, globals_):
            edge_feature_sizes = [self.latent_size] * self.hidden_layers
            if use_global_model:
                inputs = jnp.concatenate((edges, senders, receivers, globals_), axis=1)
            else:
                inputs = jnp.concatenate((edges, senders, receivers), axis=1)
            # vmap across batch dim so that the MLP processed each edge separately
            model = vmapMLP(feature_sizes=edge_feature_sizes,
                            activation=self.activation,
                            dropout_rate=self.dropout_rate, 
                            deterministic=not self.training,
                            with_layer_norm=self.layer_norm)
                            # name='update_edge')
            return model(inputs)
            
        def update_global_fn(nodes, edges, globals_):
            model = None
            if self.use_global_model:
                raise NotImplementedError('Implement global update function!')
            return model
        
        if not self.use_edge_model: update_edge_fn = None

        if not self.use_global_model: update_global_fn = None

        graph_net = jraph.GraphNetwork(
                update_node_fn=update_node_fn,
                update_edge_fn=update_edge_fn,
                update_global_fn=update_global_fn,
            )
        
        num_nets = self.num_mp_steps if not self.shared_params else 1
        graph_nets = []
        for _ in range(num_nets):
            graph_nets.append(graph_net)
        
        # Encode features to latent space
        processed_graph = encoder(graph)
        prev_graph = processed_graph

        # Message passing
        for i in range(self.num_mp_steps): 
            j = i if not self.shared_params else 0
            processed_graph = graph_nets[j](processed_graph)
            processed_graph = processed_graph._replace(nodes=processed_graph.nodes + prev_graph.nodes,
                                                       edges=processed_graph.edges + prev_graph.edges)
            prev_graph = processed_graph

        # Decode latent space features back to node features
        processed_graph = decoder(processed_graph)

        # Decoder post-processor
        processed_graph = self.decoder_postprocessor(processed_graph, aux_data)

        return processed_graph

class GNODE(nn.Module):
    """ 
        EncodeProcessDecode GNODE

        Graph Neural Network with Neural ODE message passing functions
        
        The neural ODEs takes concatenated latent features as input and its output is passed through a linear layer

        TODO:
        - fix neural ODE call - time explicitly given (from globals)?
        - inherit GraphNet?       
        - explicitly add horizon for neural ODE (integration time) 
    """
    norm_stats: FrozenConfigDict

    num_mp_steps: int = 1
    layer_norm: bool = False
    use_edge_model: bool = False
    shared_params: bool = False
    vel_history: int = 5
    horizon: int = 1

    globals_output_size: int = 0
    edge_output_size: int = 1
    node_output_size: int = 1
    prediction: str = 'acceleration'
    integration_method: str = 'semi_implicit_euler'

    # MLP parameters
    latent_size: int = 16
    hidden_layers: int = 2
    activation: str = 'relu'
    dropout_rate: float = 0
    training: bool = True

    add_self_loops: bool = False
    add_undirected_edges: bool = False

    dt: float = num_mp_steps * 0.01

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        cur_pos = graph.nodes[:,0].reshape(-1)
        cur_vel = graph.nodes[:,self.vel_history].reshape(-1)
        prev_vel = graph.nodes[:,1:self.vel_history+1]
        def update_node_fn(nodes, senders, receivers, globals_):
            input = jnp.concatenate((nodes, senders, receivers, globals_), axis=1)
            node_feature_sizes = [self.latent_size] * self.hidden_layers + [input.shape[1]]
            derivative_net = MLP(feature_sizes=node_feature_sizes, 
                                 activation=self.activation, 
                                 dropout_rate=self.dropout_rate, 
                                 deterministic=not self.training)
            model = jax.vmap(NeuralODE(derivative_net), in_axes=(0,None))
            time = globals_[0]
            integration_times = jnp.array([time, time + self.horizon]).squeeze() * self.dt
            # jax.debug.print('integration times {}', integration_times)
            output = model(input, integration_times)
            return nn.Dense(self.latent_size)(output[:,-1]) # -1 because only use ode solution at last ts

        def update_edge_fn(edges, senders, receivers, globals_):
            input = jnp.concatenate((edges, senders, receivers, globals_), axis=1)
            edge_feature_sizes = [self.latent_size] * self.hidden_layers + [input.shape[1]]
            derivative_net = MLP(feature_sizes=edge_feature_sizes, 
                                 activation=self.activation,
                                 dropout_rate=self.dropout_rate, 
                                 deterministic=not self.training)
            model = jax.vmap(NeuralODE(derivative_net), in_axes=(0,None))
            time = globals_[0]
            integration_times = jnp.array([time, time + self.horizon]).squeeze() * self.dt
            output = model(input, integration_times)
            return nn.Dense(self.latent_size)(output[:,-1]) # -1 because only use ode solution at last ts
            
        def update_global_fn(nodes, edges, globals_):
            del nodes, edges
            time = globals_[0]
            static_params = globals_[1:]
            globals_ = jnp.concatenate((jnp.array([time + self.horizon]), static_params))
            return globals_ 

       # Encoder
        encoder = jraph.GraphMapFeatures(
            embed_edge_fn=MLP(feature_sizes=[self.latent_size] * self.hidden_layers,
                              activation=self.activation,
                              with_layer_norm=self.layer_norm),
            embed_node_fn=MLP(feature_sizes=[self.latent_size] * self.hidden_layers,
                              activation=self.activation,
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

        decoder_latent_sizes = [self.latent_size] * self.hidden_layers
        # Decoder
        decoder = jraph.GraphMapFeatures(
            embed_node_fn=MLP(feature_sizes=decoder_latent_sizes + [self.node_output_size],
                              activation=self.activation),
            embed_edge_fn=MLP(feature_sizes=decoder_latent_sizes + [self.edge_output_size],
                              activation=self.activation),
        )

        def decoder_postprocessor(graph: jraph.GraphsTuple):
            # Use predicted acceleration to update node features using semi-implicit Euler integration
            next_nodes = None
            next_edges = None
            if self.integration_method == 'semi_implicit_euler':
                if self.layer_norm:
                    normalized_acc = graph.nodes.reshape(-1)
                    pred_acc = normalized_acc * self.norm_stats.acceleration.std + self.norm_stats.acceleration.mean
                else:
                    pred_acc = graph.nodes.reshape(-1)
                next_vel = cur_vel + pred_acc * self.dt
                next_pos = cur_pos + next_vel * self.dt

                next_nodes = jnp.column_stack([next_pos, prev_vel[:,1:], next_vel, pred_acc])
                next_edges = jnp.diff(next_pos).reshape(-1,1)
                
            elif self.integration_method == 'verlet':
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
        prev_graph = processed_graph
        # Message passing
        for i in range(self.num_mp_steps): 
            processed_graph = processor_nets[i](processed_graph)
            processed_graph = processed_graph._replace(nodes=processed_graph.nodes + prev_graph.nodes,
                                                       edges=processed_graph.edges + prev_graph.edges)
            prev_graph = processed_graph

        # Decode latent space features back to node features
        processed_graph = decoder(processed_graph)

        # Decoder post-processor
        processed_graph = decoder_postprocessor(processed_graph)

        return processed_graph