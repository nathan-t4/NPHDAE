import jraph
import jax
import diffrax
import flax
import flax.linen as nn
import jax.numpy as jnp

from typing import Sequence, Callable
from ml_collections import FrozenConfigDict
from utils.graph_utils import *

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
    """ 
        EncodeProcessDecode GN 
    """
    norm_stats: FrozenConfigDict

    num_mp_steps: int = 1
    layer_norm: bool = False
    use_edge_model: bool = False
    shared_params: bool = False
    vel_history: int = 5
    control_history: int = 1
    noise_std: float = 0.0003

    globals_output_size: int = 0
    edge_output_size: int = 1
    node_output_size: int = 1
    prediction: str = 'acceleration'
    integration_method: str = 'SemiImplicitEuler'
    
    # MLP parameters
    latent_size: int = 16
    hidden_layers: int = 2
    activation: str = 'relu'
    dropout_rate: float = 0
    training: bool = True

    add_self_loops: bool = False
    add_undirected_edges: bool = False

    dt: float = 0.01

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, next_u, rng) -> jraph.GraphsTuple:
        next_u = next_u[1::2] # get nonzero elements (even indices) corresponding to control input
        if self.training: 
            # Add noise to current position (first node feature)
            rng, pos_rng, u_rng = jax.random.split(rng, 3)
            pos_noise = self.noise_std * jax.random.normal(pos_rng, (len(graph.nodes),))
            new_nodes = jnp.column_stack((graph.nodes[:,0].T + pos_noise, graph.nodes[:,1:]))
            graph = graph._replace(nodes=new_nodes)
            # Add noise to control input at current time-step (next_u)
            next_u_noise = self.noise_std * jax.random.normal(u_rng, (len(next_u),))
            next_u = next_u + next_u_noise
    
        cur_pos = graph.nodes[:,0]
        if self.prediction == 'acceleration':
            cur_vel = graph.nodes[:,self.vel_history]
            prev_vel = graph.nodes[:,1:self.vel_history+1] # includes current velocity
            prev_u = graph.nodes[:,self.vel_history+1:] # includes current u
        elif self.prediction == 'position':
            prev_pos = graph.nodes[:,1:self.vel_history+1]

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
            
        def update_global_fn(nodes, edges, globals_):
            del nodes, edges
            time = globals_[0]
            static_params = globals_[1:]
            globals_ = jnp.concatenate((jnp.array([time + self.num_mp_steps]), static_params))
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

        if graph.globals is None:
            update_global_fn = None

        num_nets = self.num_mp_steps if not self.shared_params else 1
        processor_nets = []
        for _ in range(num_nets): # TODO replace with scan
            net = jraph.GraphNetwork(
                update_node_fn=update_node_fn,
                update_edge_fn=update_edge_fn,
                update_global_fn=update_global_fn,
            )
            processor_nets.append(net)

        # Decoder
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
            # if self.prediction == 'acceleration':
            def force(t, args):
                del t, args
                normalized_acc = graph.nodes
                # M = jnp.diag(masses)
                # normalized_u_c = A @ M @ normalized_acc # need norm_stats for all composed subsystems 
                # pred_uc = 
                # pred_acc = pred_acc - u_c
                pred_acc = normalized_acc * self.norm_stats.acceleration.std + self.norm_stats.acceleration.mean
                return pred_acc
            
            def newtons_equation_of_motion(t, y, args):
                """
                    TODO: generalize to n-dim systems
                    y = = [pos, vel] = [x0, x1, v0, v1] 
                """
                A = jnp.array([[0, 0, 1, 0], 
                               [0, 0, 0, 1], 
                               [0, 0, 0, 0], 
                               [0, 0, 0, 0]])
                F = jnp.concatenate((jnp.array([[0], [0]]), force(t, args)))
                return A @ y + F
            
            @jax.jit
            def solve(y0, args):
                t0 = 0
                t1 = self.num_mp_steps * self.dt
                dt0 = self.dt
                match self.integration_method:
                    case 'Euler':
                        term = diffrax.ODETerm(newtons_equation_of_motion)
                        solver = diffrax.Euler()
                        sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, args=args)
                        next_pos = sol.ys[-1, 0:2]
                        next_vel = sol.ys[-1, 2:4]
                    case 'Tsit5':
                        term = diffrax.ODETerm(newtons_equation_of_motion)
                        solver = diffrax.Tsit5()
                        sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, args=args)
                        next_vel = sol.ys[-1, 2:4]
                    case 'SemiImplicitEuler':
                        pred_acc = force(t0, args).squeeze()
                        next_vel = cur_vel + pred_acc * (dt0 * self.num_mp_steps)
                        next_pos = cur_pos + next_vel * (dt0 * self.num_mp_steps)
                    case _:
                        raise NotImplementedError('Invalid integration method')    
                return next_pos, next_vel
            
            y0 = jnp.concatenate((cur_pos, cur_vel), axis=0).reshape(-1, 1)
            args = None
            next_pos, next_vel = solve(y0, args)  
            next_nodes = jnp.column_stack((next_pos, 
                                           prev_vel[:,1:], next_vel, 
                                           prev_u[:,1:], next_u, 
                                           force(None, args))) # TODO: pass time into force?
            next_edges = jnp.diff(next_pos.squeeze()).reshape(-1,1)
            # else:
            #     raise NotImplementedError('Invalid prediction mode')
            
            if self.add_undirected_edges:
                next_edges = jnp.concatenate((next_edges, next_edges), axis=0)
            
            if self.add_self_loops:
                N = len(next_u) # num of masses
                next_edges = jnp.concatenate((next_edges, jnp.zeros((N, 1))), axis=0)
            
            if self.use_edge_model:
                graph = graph._replace(nodes=next_nodes, edges=next_edges)
            else:
                graph = graph._replace(nodes=next_nodes)   

            return graph

        # Encode features to latent space
        processed_graph = encoder(graph)
        prev_graph = processed_graph
        # Message passing
        for i in range(num_nets): 
            processed_graph = processor_nets[i](processed_graph)
            processed_graph = processed_graph._replace(nodes=processed_graph.nodes + prev_graph.nodes,
                                                       edges=processed_graph.edges + prev_graph.edges)
            prev_graph = processed_graph
        # def mp_step(graph, net):
        #     next_graph = net(graph)
        #     next_graph = next_graph._replace(nodes=next_graph.nodes + graph.nodes,
        #                                      edges=next_graph.edges + graph.edges)
        #     return next_graph, _
        # processed_graph, _ = jax.lax.scan(mp_step, processed_graph, processor_nets)
        # Decode latent space features back to node features
        processed_graph = decoder(processed_graph)

        # Decoder post-processor
        processed_graph = decoder_postprocessor(processed_graph)

        return processed_graph
    
class CoupledGraphNetworkSimulator(nn.Module):
    """ 
        Model-based composition of Graph Network Simulators
    """
    join_graph: Callable[[jraph.GraphsTuple, jraph.GraphsTuple, jnp.array], jraph.GraphsTuple]
    dejoin_graph: Callable[[jraph.GraphsTuple, jnp.array], tuple[jraph.GraphsTuple, jraph.GraphsTuple]]
    GNS_one: GraphNetworkSimulator
    GNS_two: GraphNetworkSimulator
    merged_nodes: jnp.array # shape = [#, 2]

    @nn.compact
    def __call__(
        self, composed_graph: jraph.GraphsTuple, composed_next_u: jnp.array, composed_mass: jnp.array, rng
        ) -> jraph.GraphsTuple:
        """
            || --- 0 --- 1   (+)   1 --- 2   (=)  || --- 0 --- 1 --- 2

            Steps
            1. dejoin input composed_graph into graph_one, graph_two
            2. pass respective GNS on graph_one and graph_two. get acceleration
            3. add new inputs
            4. pass graph_one and graph_two through GNS again with new_inputs 
            5. join graph and return

            # Problem: 
            - graph_one from decomposition of 5-msd does not have the same behavior as graph from 2-msd!


                       delta y
                      <-------> 
            ||---0---1  o---o  1---2

            TODO
            - add delta y (maybe just when plotting simulation)
            - learn u_c?
        """
        graph_one, graph_two = self.dejoin_graph(composed_graph, self.merged_nodes)

        idx_1 = jnp.array([0, 1]) # TODO: given merged nodes, original nodes of 1 (first N1) and 2 (next N2)
        idx_2 = jnp.array([1, 2])
        next_u_1 = composed_next_u[idx_1]
        next_u_2 = composed_next_u[idx_2]

        jax.debug.print('u_1 {} u_2 {}', next_u_1, next_u_2)

        next_graph_one = self.GNS_one(graph_one, next_u_1, rng)
        next_graph_two = self.GNS_two(graph_two, next_u_2, rng)

        # nominal normalized accelerations
        nominal_acc_one = next_graph_one.nodes[:,-1]
        nominal_acc_two = next_graph_two.nodes[:,-1]

        # rescale accelerations
        acc_one = nominal_acc_one * self.GNS_one.norm_stats.acceleration.std + self.GNS_one.norm_stats.acceleration.mean
        acc_two = nominal_acc_two * self.GNS_two.norm_stats.acceleration.std + self.GNS_two.norm_stats.acceleration.mean

        jax.debug.print('acc one {} and acc two {}', acc_one, acc_two)

        # Assume masses are known. TODO: remove assumption
        M1 = jnp.diag(composed_mass[idx_1]) 
        M2 = jnp.diag(composed_mass[idx_2])
        # Calculate new forces due to joining
        u_c_1 = jnp.array([0, M1 @ acc_two[0]])
        u_c_2 = jnp.array([M2 @ acc_one[1], 0])
        # Get control inputs for composed system # TODO: use self.merged_nodes
        F_1 = next_u_1 + u_c_1
        F_2 = next_u_2 + u_c_2

        jax.debug.print('F_1 {} F_2 {}', F_1, F_2)

        # predict composed system dynamics
        next_graph_one = self.GNS_one(graph_one, F_1, rng)
        next_graph_two = self.GNS_two(graph_two, F_2, rng)

        # transform features of next_graph_two (merging flows) - adding position!
        next_graph_two_transformed_position = next_graph_two.nodes[:,0] + next_graph_one[-1,0] # pos of merged node
        next_graph_two._replace(nodes=jnp.concatenate((next_graph_two_transformed_position,
                                                       next_graph_two.nodes[:,1])),
                                                       axis=1)

        next_graph = self.join_graph(next_graph_one, next_graph_two, self.merged_nodes)
        
        return next_graph

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