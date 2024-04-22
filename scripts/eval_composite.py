import os
import jax
import optax
import json

import jax.numpy as jnp

import ml_collections
from clu import metric_writers, periodic_actions, checkpoint

from time import strftime
from timeit import default_timer
from flax.training.train_state import TrainState
from graph_builder import MSDGraphBuilder
from scripts.models import *
from utils.gnn_utils import *
from utils.data_utils import *
from utils.jax_utils import *
from config import *

def dejoin_graph_example(
        composed_graph: jraph.GraphsTuple,
        merged_nodes: jnp.array,
    ) -> tuple[jraph.GraphsTuple, jraph.GraphsTuple]:
    """ 
        For (4 = 2+2) mass spring example

        || --- 0 --- 1   (+)   1 --- 2   (=)  || --- 0 --- 1 --- 2
        
        Indices are hard-coded
    """    
    # number of vertices and edges of subsystem 1
    NV1 = 2
    NE1 = 1
    indices_1 = jnp.array([0,2,4,5])
    # number of vertices and edges of subsystem 2
    NV2 = 2
    NE2 = 1
    indices_2 = jnp.array([1,3,5,6])

    nodes_one = composed_graph.nodes[jnp.array([0, 1]),:]
    edges_one = composed_graph.edges[jnp.array([0]),:]
    receivers_one = composed_graph.receivers[indices_1]
    senders_one = composed_graph.senders[indices_1]

    nodes_two = composed_graph.nodes[jnp.array([1, 2]),:]
    edges_two = composed_graph.edges[jnp.array([1]),:]
    receivers_two = composed_graph.receivers[indices_2]
    senders_two = composed_graph.senders[indices_2]

    graph_one = jraph.GraphsTuple(
        nodes=nodes_one,
        edges=edges_one,
        globals=None,
        receivers=receivers_one,
        senders=senders_one,
        n_node=jnp.array([NV1]),
        n_edge=jnp.array([NE1]),
    )

    graph_two = jraph.GraphsTuple(
        nodes=nodes_two,
        edges=edges_two,
        globals=None,
        receivers=receivers_two,
        senders=senders_two,
        n_node=jnp.array([NV2]),
        n_edge=jnp.array([NE2]),
    )

    return graph_one, graph_two

def join_graph_example(
        graph_one: jraph.GraphsTuple, 
        graph_two: jraph.GraphsTuple,
        merged_nodes: jnp.array,
    ) -> jraph.GraphsTuple:
    nodes_two_indices_without_overlap = jnp.array([1])
    two_indices_without_overlap = jnp.array([0, 1, 3])
    # The next line chooses to use GNS results from GNS_1 for overlapping node 1
    nodes = jnp.concatenate((graph_one.nodes, graph_two.nodes[nodes_two_indices_without_overlap, :]))
    edges = jnp.concatenate((graph_one.edges, graph_two.edges)) # edge sets are disjoint
    receivers = jnp.concatenate((graph_one.receivers, graph_two.receivers[two_indices_without_overlap]))
    senders = jnp.concatenate((graph_one.senders, graph_two.senders[two_indices_without_overlap]))
    # globals = jnp.concatenate((graph_one.globals, graph_two.globals)) # since globals is None for both
    n_node = graph_one.n_node + graph_two.n_node - 1 # one merged node
    n_edge = graph_one.n_edge + graph_two.n_edge

    composed_graph = jraph.GraphsTuple(
        nodes = nodes,
        edges=edges,
        receivers=receivers,
        senders=senders,
        globals=None,
        n_node=n_node,
        n_edge=n_edge
    )

    return composed_graph

def eval_composite_GNS(config: ml_collections.ConfigDict):
    # load gns1 from checkpoint
    # load gns2 from checkpoint
    # compare with actual (gt acceleration)
    # compare with expected data AND one-shot GNS
    paths = config.paths
    eval_params = config.eval_params    
    net_params_one = config.net_params_one
    net_params_two = config.net_params_two

    dir = os.path.join(os.curdir, f'results/GNS/compose_gnn/{strftime("%m%d-%H%M%S")}_{eval_params.trial_name}')

    assert (net_params_one.prediction == net_params_two.prediction), \
        'Prediction modes should be the same for both GNS'
    prediction = net_params_one.prediction

    assert (net_params_one.add_undirected_edges == net_params_two.add_undirected_edges), \
        'Undirected edges should be the same for both GNS'
    undirected_edges = net_params_one.add_undirected_edges

    assert (net_params_one.add_self_loops == net_params_two.add_self_loops), \
        'Self loops should be the same for both GNS'
    self_loops = net_params_one.add_self_loops

    assert (net_params_one.vel_history == net_params_two.vel_history), \
        'Velocity history should be the same for both GNS'
    vel_history = net_params_one.vel_history

    assert (net_params_one.control_history == net_params_two.control_history), \
        'Control history should be the same for both GNS'
    control_history = net_params_one.control_history

    assert (net_params_one.num_mp_steps == net_params_two.num_mp_steps), \
        'Number of message passing steps should be the same for both GNS'
    num_mp_steps = net_params_one.num_mp_steps

    def create_net(net_params):
        return GraphNetworkSimulator(**net_params)
            
    log_dir = os.path.join(dir, 'log')
    plot_dir = os.path.join(dir, 'eval_plots')

    checkpoint_dir_one = os.path.join(paths.dir_one, 'checkpoint')
    checkpoint_dir_two = os.path.join(paths.dir_two, 'checkpoint')

    rng = jax.random.key(0)
    rng, init_rng, net_rng = jax.random.split(rng, 3)

    # Create writer for logs
    writer = metric_writers.create_default_writer(logdir=log_dir)

    # Create optimizers
    tx = optax.adam(**config.optimizer_params)

    # Create graph builders for subsystems 1, 2, and the composite system
    gb_one = MSDGraphBuilder(paths.evaluation_data_path_one, 
                             undirected_edges, 
                             self_loops, 
                             prediction, 
                             vel_history,
                             control_history)
    
    gb_two = MSDGraphBuilder(paths.evaluation_data_path_two, 
                             undirected_edges, 
                             self_loops, 
                             prediction, 
                             vel_history,
                             control_history)

    gb_composed = MSDGraphBuilder(paths.evaluation_data_path_comp,
                                  undirected_edges,
                                  self_loops,
                                  prediction,
                                  vel_history,
                                  control_history)
            
    # Initialize first GNS
    net_params_one.norm_stats = gb_one._norm_stats
    eval_net_one = create_net(net_params_one)
    eval_net_one.training = False

    init_graph = gb_one.get_graph(traj_idx=0, t=vel_history+1)
    init_control = gb_one._control[0, 0]
    params_one = eval_net_one.init(init_rng, 
                                   init_graph,
                                   init_control, 
                                   net_rng)
    tx_one = optax.adam(**config.optimizer_params)
    batched_apply_one = jax.vmap(eval_net_one.apply, in_axes=(None,0,0,None))
    state_one = TrainState.create(
        apply_fn=batched_apply_one,
        params=params_one,
        tx=tx_one,
    )

    # Load first GNS
    checkpoint_dir_one = os.path.join(checkpoint_dir_one, 'best_model')
    ckpt_one = checkpoint.Checkpoint(checkpoint_dir_one)
    state_one = ckpt_one.restore_or_initialize(state_one)

    # Initialize second GNS
    net_params_two.norm_stats = gb_two._norm_stats
    eval_net_two = create_net(net_params_two)
    eval_net_two.training = False

    init_graph = gb_one.get_graph(traj_idx=0, t=vel_history+1)
    init_control = gb_one._control[0, 0]
    params_two = eval_net_two.init(init_rng, 
                                   init_graph,
                                   init_control, 
                                   net_rng)
    tx_two = optax.adam(**config.optimizer_params)
    batched_apply_two = jax.vmap(eval_net_two.apply, in_axes=(None,0,0,None))
    state_two = TrainState.create(
        apply_fn=batched_apply_two,
        params=params_two,
        tx=tx_two,
    )

    # Load second GNS
    checkpoint_dir_two = os.path.join(checkpoint_dir_two, 'best_model')
    ckpt_two = checkpoint.Checkpoint(checkpoint_dir_two)
    state_two = ckpt_two.restore_or_initialize(state_two)

    # hard-coded nodes to merge
    merged_nodes = jnp.array([1, 0])

    # Initialized composite GNS
    eval_net_composed = CoupledGraphNetworkSimulator(join_graph=join_graph_example,
                                                     dejoin_graph=dejoin_graph_example,
                                                     GNS_one=eval_net_one,
                                                     GNS_two=eval_net_two,
                                                     merged_nodes=merged_nodes)
    init_graph = gb_composed.get_graph(traj_idx=0, t=vel_history+1)
    init_control = gb_composed._control[0, 0]
    init_mass = gb_composed._m[0]

    params = eval_net_composed.init(init_rng, 
                                    init_graph,
                                    init_control, 
                                    init_mass,
                                    net_rng)
    batched_apply = jax.vmap(eval_net_composed.apply, in_axes=(None,0,0,0,None))
    eval_state = TrainState.create(
        apply_fn=batched_apply,
        params=params,
        tx=tx,
    )

    def rollout(state: TrainState, traj_idx: int = 0, t0: int = 0):
        """
            rollout 
            - get graph
        """
        tf_idxs = (t0 + jnp.arange(eval_params.rollout_timesteps // num_mp_steps)) * num_mp_steps
        t0 = round(vel_history /  num_mp_steps) * num_mp_steps # min of one and two?
        tf_idxs = jnp.unique(tf_idxs.clip(min=t0 + num_mp_steps, max=1500))
        ts = tf_idxs * eval_net_one.dt
        
        masses = jnp.tile(gb_composed._m[traj_idx], (len(tf_idxs),1))
        controls = gb_composed._control[traj_idx, tf_idxs]

        exp_qs_buffer = gb_composed._qs[traj_idx, tf_idxs]
        exp_as_buffer = gb_composed._accs[traj_idx, tf_idxs]
        graphs = gb_composed.get_graph(traj_idx, t0)
        batched_graphs = pytrees_stack([graphs])

        def forward_pass(graph, aux_data):
            control, mass = aux_data
            graph = state.apply_fn(
                state.params, graph, jnp.array([control]), jnp.array([mass]), jax.random.key(0)
            )
            pred_qs = graph.nodes[:,:,0]

            if prediction == 'acceleration': # change to config 
                pred_accs = graph.nodes[:,:,-1]
                # remove acceleration  
                graph = graph._replace(nodes=graph.nodes[:,:,:-1])
                
                return graph, (pred_qs.squeeze(), pred_accs.squeeze())

        start = default_timer()
        final_batched_graph, pred_data = jax.lax.scan(forward_pass, batched_graphs, (controls, masses))
        end = default_timer()
        jax.debug.print('Inference time {} [sec] for {} passes', end - start, len(ts))

        eval_pos_loss = optax.l2_loss(predictions=pred_data[0], targets=exp_qs_buffer).mean()

        if prediction == 'acceleration':
            aux_data = (gb_composed._m[traj_idx], gb_composed._k[traj_idx], gb_composed._b[traj_idx])
            return ts, np.array(pred_data), np.array((exp_qs_buffer, exp_as_buffer)), aux_data, EvalMetrics.single_from_model_output(loss=eval_pos_loss)

    print(f"Number of parameters {num_parameters(params)}")
    rollout_error_sum = 0
    for i in range(len(gb_composed._data)):
        ts, pred_data, exp_data, aux_data, eval_metrics = rollout(eval_state, traj_idx=i)
        writer.write_scalars(
            i, add_prefix_to_keys(eval_metrics.compute(), f'eval {paths.evaluation_data_path_comp}')
        )
        rollout_error_sum += eval_metrics.compute()['loss']
        plot_evaluation_curves(ts, pred_data, exp_data, aux_data, plot_dir=plot_dir, prefix=f'eval_traj_{i}')

    print('Mean rollout error: ', rollout_error_sum / len(gb_composed._data))

    # Save evaluation metrics to json
    eval_metrics = {
        'mean_rollout_error': (rollout_error_sum / len(gb_composed._data)).tolist()
    }
    eval_metrics_file = os.path.join(plot_dir, 'eval_metrics.js')
    with open(eval_metrics_file, "w") as outfile:
        json.dump(eval_metrics, outfile)

if __name__ == '__main__':
    config = create_comp_gnn_config()
    eval_composite_GNS(config)