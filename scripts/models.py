import jraph
import jax
import diffrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from typing import Sequence, Callable
from flax.training.train_state import TrainState
from ml_collections import FrozenConfigDict
from utils.graph_utils import *
from utils.jax_utils import *
from utils.models_utils import *

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
    
class CustomEdgeGraphNetworkSimulator(nn.Module):
    """
        Graph Network Simulator with two edge decoders. 
        
        Edges with indices "edge_idxs" are processed with the first decoder, 
        and all other edges are processed with the second decoder
    """
    edge_idxs: list
    encoder_node_fn: Callable
    encoder_edge_fns: Sequence[Callable]
    decoder_node_fn: Callable
    decoder_edge_fns: Sequence[Callable]
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
        def CustomEdgeGraphMapFeatures(embed_edge_fn_1 = None,
                                       embed_edge_fn_2 = None,
                                       embed_node_fn = None,
                                       embed_global_fn = None,
                                       embed_edge_fns = Sequence[Callable]):
            identity = lambda x : x
            # TODO:
            # for i in range(len(embed_edge_fns)):
            #     embed_edge_fns[i] = embed_edge_fns[i] if embed_edge_fns[i] else identity

            # embed_edges_fn_1 = embed_edge_fn_1 if embed_edge_fn_1 else identity
            # embed_edges_fn_2 = embed_edge_fn_2 if embed_edge_fn_2 else identity
            embed_nodes_fn = embed_node_fn if embed_node_fn else identity
            embed_globals_fn = embed_global_fn if embed_global_fn else identity

            def Embed(graph):
                """
                1. differentiate edges_one and edges_two
                2. apply embed_edge_fn_1 on edges_one and embed_edge_fn_2 on edges_two (in place?)
                3. replace graph edges with new edges, nodes with new nodes, and globals with new globals
                4. return graph
                """
                new_edges = None
                for i in range(len(graph.edges)):
                    edge_type = np.where(self.edge_idxs == i)
                    # Check if i is not in self.edge_idxs
                    edge_type = None if edge_type[0].shape[0] == 0 else edge_type[0]
                    if edge_type is not None:
                        new_edge = embed_edge_fns[edge_type.item()](graph.edges[i])
                    else:
                        new_edge = embed_edge_fns[-1](graph.edges[i])
                    
                    # if i in self.edge_idxs[0]:
                    #     new_edge = embed_edges_fn_1(graph.edges[i])
                    # else:
                    #     new_edge = embed_edges_fn_2(graph.edges[i])
                    
                    if new_edges is None:
                        new_edges = new_edge
                    else:
                        new_edges = jnp.concatenate((new_edges, new_edge))

                return graph._replace(nodes=embed_nodes_fn(graph.nodes),
                                      edges=new_edges.reshape(graph.edges.shape[0], -1  ),
                                      globals=embed_globals_fn(graph.globals))    
            return Embed
        
        encoder = CustomEdgeGraphMapFeatures(
            embed_node_fn=self.encoder_node_fn,
            embed_edge_fns=self.encoder_edge_fns,
        )  

        decoder = CustomEdgeGraphMapFeatures(
            embed_node_fn=self.decoder_node_fn,
            embed_edge_fns=self.decoder_edge_fns,
        )

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

class CompGraphNetworkSimulator(nn.Module):
    """ 
        Model-based composition of Graph Network Simulators
    """
    latent_size: int
    hidden_layers: int
    activation: str

    I: jnp.array # interconnection structure

    split_u: Callable[[jnp.array, jnp.array], tuple[jnp.array, jnp.array]]
    split_graph: Callable[[jraph.GraphsTuple, jnp.array], tuple[jraph.GraphsTuple, jraph.GraphsTuple]]
    join_acc: Callable[[jnp.array, jnp.array, jnp.array], jnp.array]
    join_graph: Callable[[jraph.GraphsTuple, jraph.GraphsTuple, jnp.array], jraph.GraphsTuple]

    state_one: TrainState
    state_two: TrainState
    
    @nn.compact
    def __call__(
        self, composed_graph: jraph.GraphsTuple, u_c: jnp.array, rng: jax.Array,
        ) -> jraph.GraphsTuple:
        """
            || --- 0 --- 1   (+)   1 --- 2   (=)  || --- 0 --- 1 --- 2

            # graph_one, graph_two = split(composed_graph, I)
            # F = f(composed_graph)
            # u = u_c + F
            # u_1, u_2 = split(u, I)
            # next_graph_one = self.GNS_one(graph_one, u_1, rng)
            # next_graph_two = self.GNS_two(graph_two, u_2, rng)
            # acc_one = next_graph_one.nodes[:,-1]
            # acc_two = next_graph_two.nodes[:,-1]
            # acc_est = join(acc_one, acc_two, I)
            # next_graph = join(graph_one, graph_two, I)

                       delta y
                      <-------> 
            ||---0---1  o---o  1---2

        """
        N = len(composed_graph.nodes)
        # TODO: generalize f to be a GraphNetwork?
        f = MLP(feature_sizes=[self.latent_size] * self.hidden_layers + [N], activation=self.activation)
        # jax.debug.print('graph c nodes {} edges {} globals {} receivers {} senders {} n_node {} n_edge {}', composed_graph.nodes.shape, composed_graph.edges.shape, composed_graph.globals, composed_graph.receivers, composed_graph.senders, composed_graph.n_node, composed_graph.n_edge)
        graph_one, graph_two = self.split_graph(composed_graph, self.I)
        # jax.debug.print('graph two nodes {} edges {} globals {} receivers {} senders {} n_node {} n_edge {}', graph_two.nodes.shape, graph_two.edges.shape, graph_two.globals, graph_two.receivers, graph_two.senders, graph_two.n_node, graph_two.n_edge)
        # Forces from joining systems
        # F = f(composed_graph) # TODO: when GraphNetwork
        inputs = jnp.concatenate((graph_one.nodes, graph_two.nodes), axis=0).flatten()
        F = [0] * (2*N)
        F[1::2] = f(inputs)
        F = jnp.array(F)
        # Updated control input for composed system
        u = u_c + F # TODO: this is not correct. should add F on nodes depending on I
        u_1, u_2 = self.split_u(u, self.I) # TODO: should take u_c and F as input?

        # Get next state - need to stack then unstack because state apply_fn uses vmap
        graph_one = pytrees_stack([graph_one])
        graph_two = pytrees_stack([graph_two])

        next_graph_one = self.state_one.apply_fn(self.state_one.params, graph_one, jnp.array([u_1]), rng)
        next_graph_two = self.state_two.apply_fn(self.state_two.params, graph_two, jnp.array([u_2]), rng)

        next_graph_one = pytrees_unstack(graph_one)
        next_graph_two = pytrees_unstack(graph_two)

        # jax.debug.print('next graph one nodes shape {}', next_graph_one.nodes.shape)

        # Predicted accelerations from GNS
        acc_one = next_graph_one.nodes[:,-1]
        acc_two = next_graph_two.nodes[:,-1]
        
        # Acceleration of composed system
        # jax.debug.print('acc one {} acc two {}', acc_one, acc_two)
        acc_est = self.join_acc(acc_one, acc_two, self.I)
        # jax.debug.print('acc est {}', acc_est)
        # jax.debug.print('next graph two nodes {}', next_graph_two.nodes)

        # Add position offset to next_graph_two (TODO: remove)
        next_graph_two_pos = next_graph_two.nodes[:,0] + next_graph_one.nodes[-1, 0]
        next_graph_two_nodes = jnp.concatenate((
            next_graph_two_pos.reshape((-1,1)),
            next_graph_two.nodes[:,1:]
        ), axis=1) # plus delta_y

        # jax.debug.print('next graph two nodes {}', next_graph_two_nodes)

        next_graph_two = next_graph_two._replace(nodes=next_graph_two_nodes)
        # Join graphs
        next_composed_graph = self.join_graph(next_graph_one, next_graph_two, self.I)
        # Add acc_est to nodes of next_composed_graph 
        nodes = jnp.column_stack((next_composed_graph.nodes, acc_est))

        next_composed_graph = next_composed_graph._replace(nodes=nodes)

        # jax.debug.print('next graph nodes {} edges {} receivers {} senders {}', next_composed_graph.nodes, next_composed_graph.edges, next_composed_graph.receivers, next_composed_graph.senders)

        return next_composed_graph

class CompGNS(nn.Module):
    state_one: TrainState
    state_two: TrainState
    I: jnp.array # interconnection structure

    @nn.compact
    def __call__(self, graph_one, graph_two, u_one, u_two, rng):
        """
            Given any initial composite graph (outside of this method) that is split into two subsystem graphs (also outside of this method), predict the next-state of the composite system.

            The initial composite graph is split to get the correct initial conditions...

            Input: G1, G2
            Input: u_1, u_2 (if input on same node, then what to do?)
            Input: state_one, state_two (GNS)

            TODO: need to make sure GNS is position invariant
            TODO: recreate composition idea on mass springs?
        """
        graph_one_acc = self.state_one.apply_fn(self.state_one.params, graph_one, u_one, rng)
        graph_two_acc = self.state_two.apply_fn(self.state_two.params, graph_two, u_two, rng)

        acc_one = graph_one_acc.nodes[:,-1]
        acc_two = graph_two_acc.nodes[:,-1]

        N_c = len(graph_one.nodes) + len(graph_two.nodes) - len(self.I)
        """
            i=0,1,2

            V_1 = {0,1}
            V_2 = {1,2}

            0 \in V_1 / V_m
            1 \in V_m
            2 \in V_2 / V_m

            I = 
            {
                1: [1, 0],
            }
        """
        F_ext = np.zeros((N_c, 2))
        for k, v in self.I.values():
            F_ext[k] = [acc_one[v[0]], acc_two[v[1]]]
        
        next_graph_one = self.state_one.apply_fn(self.state_one.params, graph_one, u_one + F_ext[:,0])
        next_graph_two = self.state_two.apply_fn(self.state_two.params, graph_two, u_two + F_ext[:,1])

        next_x_c = get_state(next_graph_one, next_graph_two) # TODO: because the state at the merged nodes needs to be dealt with special care

        return next_graph_one, next_graph_two, next_x_c
    
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