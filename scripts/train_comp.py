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
from flax.training.early_stopping import EarlyStopping

from graph_builder import MSDGraphBuilder
from scripts.models import *
from utils.gnn_utils import *
from utils.data_utils import *
from utils.jax_utils import *
from config import *

def split_u_ex(u, I):
    """ TODO - generalize with I """
    return u[:4], u[2:]

def split_graph_ex(Gc, I):
    """ TODO - generalize with I """
    indices_1 = jnp.array([0,2,4,5])
    indices_2 = jnp.array([1,3,5,6])

    n1 = Gc.nodes[jnp.array([0, 1]),:]
    e1 = Gc.edges[indices_1,:]
    r1 = Gc.receivers[indices_1]
    s1 = Gc.senders[indices_1]

    n2 = Gc.nodes[jnp.array([1, 2]),:]
    e2 = Gc.edges[indices_2,:]
    r2 = Gc.receivers[indices_2]
    s2 = Gc.senders[indices_2]

    G1 = jraph.GraphsTuple(nodes=n1,
                           edges=e1,
                           globals=None,
                           receivers=r1,
                           senders=s1,
                           n_node=jnp.array([len(n1)]),
                           n_edge=jnp.array([len(e1)]))
    G2 = jraph.GraphsTuple(nodes=n2,
                           edges=e2,
                           globals=None,
                           receivers=r2,
                           senders=s2,
                           n_node=jnp.array([len(n2)]),
                           n_edge=jnp.array([len(e2)]))
    return G1, G2

def join_acc_ex(a1, a2, I):
    """ TODO - generalize with I """
    return jnp.concatenate((a1, jnp.array([a2[1]])))# TODO: this is a choice, could also be jnp.concatenate(jnp.array([a1[0]]), a2))

def join_graph_ex(G1, G2, I):
    """ TODO - generalize with I """
    overlap_ind = jnp.array([0, 1, 3])
    n = jnp.concatenate((G1.nodes, G2.nodes[1].reshape((1,-1))), axis=0)
    e = jnp.concatenate((G1.edges, G2.edges), axis=0) # TODO: remove overlap edge - identity on 1.
    r = jnp.concatenate((G1.receivers, G2.receivers[overlap_ind]), axis=0) # TODO: check
    s = jnp.concatenate((G1.senders, G2.senders[overlap_ind]), axis=0) # TODO: check
    Gc = jraph.GraphsTuple(nodes=n,
                           edges=e,
                           globals=None,
                           receivers=r,
                           senders=s,
                           n_node=jnp.array([len(n)]),
                           n_edge=jnp.array([len(e)]))
    return Gc

def dejoin_graph_example(
        composed_graph: jraph.GraphsTuple,
        merged_nodes: jnp.array,
    ) -> tuple[jraph.GraphsTuple, jraph.GraphsTuple]:
    """ 
        For (3 = 2+2) mass spring example

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

def train_comp(config: ml_collections.ConfigDict):
    # Inputs: GraphBuilder1, GraphBuilder2
    # Inputs: GNS1, GNS2
    # Inputs: control inputs of the composed system
    # Inputs: state of composed system at time-step k
    # Output: state of composed system at time-step k+1


    paths = config.paths
    training_params = config.training_params
    net_params_one = config.net_params_one
    net_params_two = config.net_params_two
    net_params_c = config.net_params_c

    net_params_c.split_u = split_u_ex
    net_params_c.split_graph = split_graph_ex
    net_params_c.join_acc = join_acc_ex
    net_params_c.join_graph = join_graph_ex
    net_params_c.I = jnp.array([]) # TODO

    if paths.dir == None:
        config.paths.dir = os.path.join(os.curdir, f'results/{training_params.net_name}/{strftime("%m%d-%H%M")}_{training_params.trial_name}')
        paths.dir = config.paths.dir
            
    log_dir = os.path.join(paths.dir, 'log')
    checkpoint_dir = os.path.join(paths.dir, 'checkpoint')
    plot_dir = os.path.join(paths.dir, 'plots')

    def assertions():
        assert (net_params_one.prediction == net_params_two.prediction), \
            'Prediction modes should be the same for both GNS'
        
        assert (net_params_one.add_undirected_edges == net_params_two.add_undirected_edges), \
            'Undirected edges should be the same for both GNS'

        assert (net_params_one.add_self_loops == net_params_two.add_self_loops), \
            'Self loops should be the same for both GNS'

        assert (net_params_one.vel_history == net_params_two.vel_history), \
            'Velocity history should be the same for both GNS'

        assert (net_params_one.control_history == net_params_two.control_history), \
            'Control history should be the same for both GNS'

        assert (net_params_one.num_mp_steps == net_params_two.num_mp_steps), \
            'Number of message passing steps should be the same for both GNS'
        
    assertions()

    prediction = net_params_one.prediction
    undirected_edges = net_params_one.add_undirected_edges
    self_loops = net_params_one.add_self_loops
    vel_history = net_params_one.vel_history
    control_history = net_params_one.control_history
    num_mp_steps = net_params_one.num_mp_steps

    def create_net(net_params):
        return GraphNetworkSimulator(**net_params)
            
    checkpoint_dir_one = os.path.join(paths.dir_one, 'checkpoint')
    checkpoint_dir_two = os.path.join(paths.dir_two, 'checkpoint')

    rng = jax.random.key(0)
    rng, init_rng, net_rng = jax.random.split(rng, 3)

    # Create writer for logs
    writer = metric_writers.create_default_writer(logdir=log_dir)

    # Create graph builders for subsystems 1, 2 (needed to initialize subsystem GNSs)
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
    
    gb_c = MSDGraphBuilder(paths.evaluation_data_path_comp,
                                undirected_edges,
                                self_loops,
                                prediction,
                                vel_history,
                                control_history)
    
    assert (gb_one._dt == gb_two._dt == gb_c._dt), \
        'dt should be the same for all GNS'
    dt = gb_one._dt
            
    # Initialize first GNS
    net_params_one.norm_stats = gb_one._norm_stats
    net_one = create_net(net_params_one)
    net_one.training = False

    init_graph_one = gb_one.get_graph(traj_idx=0, t=vel_history+1)
    init_control_one = gb_one._control[0, vel_history+1]
    params_one = net_one.init(init_rng, 
                              init_graph_one,
                              init_control_one, 
                              net_rng)
    tx_one = optax.adam(**config.optimizer_params)
    batched_apply_one = jax.vmap(net_one.apply, in_axes=(None,0,0,None)) # TODO: remove batch
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
    net_two = create_net(net_params_two)
    net_two.training = False

    init_graph_two = gb_two.get_graph(traj_idx=0, t=vel_history+1)
    init_control_two = gb_two._control[0, vel_history+1]
    params_two = net_two.init(init_rng, 
                              init_graph_two,
                              init_control_two, 
                              net_rng)
    tx_two = optax.adam(**config.optimizer_params)
    batched_apply_two = jax.vmap(net_two.apply, in_axes=(None,0,0,None)) # TODO: remove batch
    state_two = TrainState.create(
        apply_fn=batched_apply_two,
        params=params_two,
        tx=tx_two,
    )

    # Load second GNS
    checkpoint_dir_two = os.path.join(checkpoint_dir_two, 'best_model')
    ckpt_two = checkpoint.Checkpoint(checkpoint_dir_two)
    state_two = ckpt_two.restore_or_initialize(state_two)

    # Add GNS states to net_params_c
    net_params_c.state_one = state_one 
    net_params_c.state_two = state_two

    # Initialize composite GNS
    comp_net = CompGraphNetworkSimulator(**net_params_c)
    init_graph_c = gb_c.get_graph(traj_idx=0, t=vel_history+1)
    init_control_c = gb_c._control[0, vel_history+1]
    params = comp_net.init(init_rng, 
                           init_graph_c,
                           init_control_c, 
                           net_rng)
    tx = optax.adam(**config.optimizer_params)
    batched_apply = jax.vmap(comp_net.apply, in_axes=(None,0,0,None))

    state_c = TrainState.create(
        apply_fn=batched_apply,
        params=params,
        tx=tx,
    )

    # Create evaluation composite GNS
    eval_comp_net = CompGraphNetworkSimulator(**net_params_c)
    # eval_net.training = False # TODO: MLP has training and evaluation modes!
    batched_eval_apply = jax.vmap(eval_comp_net.apply, in_axes=(None,0,0,None))
    eval_state_c = state_c.replace(apply_fn=batched_eval_apply)

    def random_batch(batch_size: int, min: int, max: int, rng: jax.Array):
        """ Return random permutation of jnp.arange(min, max) in batches of batch_size """
        steps_per_epoch = (max - min)// batch_size
        perms = jax.random.permutation(rng, max - min)
        perms = perms[: steps_per_epoch * batch_size].reshape(-1,batch_size)
        return perms

    def train_epoch(state: TrainState, batch_size: int, rng: jax.Array):
        traj_perms = random_batch(batch_size, 0, gb_c._num_trajectories, rng)
        t0_perms = random_batch(batch_size, vel_history, gb_c._num_timesteps-num_mp_steps, rng) 
        dropout_rng = jax.random.split(rng, batch_size)
        rng, net_rng = jax.random.split(rng)

        def loss_fn(params, batch_graphs, batch_data):
            if training_params.loss_function == 'acceleration':
                batch_targets, batch_control = batch_data
                pred_graphs = state.apply_fn(params, batch_graphs, batch_control, net_rng, rngs={'dropout': dropout_rng})
                predictions = pred_graphs.nodes[:,:,-1] 
                loss = int(1e6) * optax.l2_loss(predictions=predictions, targets=batch_targets).mean()
            return loss

        def train_batch(state, trajs, t0s):
            tfs = t0s + num_mp_steps
            batch_control = gb_c._control[trajs, tfs]
            if training_params.loss_function == 'acceleration':
                batch_accs = gb_c._accs[trajs, tfs]
                batch_data = (batch_accs, batch_control)
            graphs = gb_c.get_graph_batch(trajs, t0s)
            # batch_graph = pytrees_stack(graphs) # explicitly batch graphs
            loss, grads = jax.value_and_grad(loss_fn)(state.params, graphs, batch_data)
            state = state.apply_gradients(grads=grads)
            return state, loss
        
        state, epoch_loss = double_scan(train_batch, state, traj_perms, t0_perms)

        train_loss = jnp.asarray(epoch_loss).mean()

        return state, TrainMetrics.single_from_model_output(loss=train_loss)

    def rollout(state: TrainState, traj_idx: int = 0, t0: int = 0):
        tf_idxs = (t0 + jnp.arange(training_params.rollout_timesteps // num_mp_steps)) * num_mp_steps
        t0 = round(vel_history /  num_mp_steps) * num_mp_steps
        tf_idxs = jnp.unique(tf_idxs.clip(min=t0 + num_mp_steps, max=gb_c._num_timesteps))
        ts = tf_idxs * dt # 
        # Get ground truth control, qs, accs
        controls = gb_c._control[traj_idx, tf_idxs]
        exp_qs_buffer = gb_c._qs[traj_idx, tf_idxs]
        exp_as_buffer = gb_c._accs[traj_idx, tf_idxs]
        # Get initial composed graph
        graphs = gb_c.get_graph(traj_idx, t0)
        batched_graphs = pytrees_stack([graphs])

        def forward_pass(graph, aux_data):
            control = aux_data
            graph = state.apply_fn(
                state.params, graph, jnp.array([control]), jax.random.key(0)
            )
            pred_qs = graph.nodes[:,:,0]

            if prediction == 'acceleration': # change to config 
                pred_accs = graph.nodes[:, :,-1]
                # remove acceleration  
                graph = graph._replace(nodes=graph.nodes[:,:,:-1])
                
                return graph, (pred_qs.squeeze(), pred_accs.squeeze())

        start = default_timer()
        final_batched_graph, pred_data = jax.lax.scan(forward_pass, batched_graphs, controls)
        end = default_timer()
        # jax.debug.print('Inference time {} [sec] for {} passes', end - start, len(ts))

        eval_pos_loss = optax.l2_loss(predictions=pred_data[0], targets=exp_qs_buffer).mean()

        if prediction == 'acceleration':
            aux_data = (gb_c._m[traj_idx], gb_c._k[traj_idx], gb_c._b[traj_idx])
            return ts, np.array(pred_data), np.array((exp_qs_buffer, exp_as_buffer)), aux_data, EvalMetrics.single_from_model_output(loss=eval_pos_loss)
    
    # Create logger to report training progress
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=training_params.num_epochs,
        writer=writer
    )
    # Load previous checkpoint (if applicable)
    ckpt = checkpoint.Checkpoint(checkpoint_dir)
    best_model_ckpt = checkpoint.Checkpoint(os.path.join(checkpoint_dir, 'best_model'))
    state_c = best_model_ckpt.restore_or_initialize(state_c)

    trajs_size = gb_c._num_trajectories
    ts_size = (gb_c._num_timesteps - gb_c._vel_history) // gb_c._dt
    steps_per_epoch = (ts_size * trajs_size) // (training_params.batch_size ** 2)
    train_fn = train_epoch

    # Setup training epochs
    init_epoch = int(state_c.step // steps_per_epoch) + 1
    final_epoch = init_epoch + training_params.num_epochs
    training_params.num_epochs = final_epoch

    early_stop = EarlyStopping(min_delta=1e-3, patience=2)

    train_metrics = None
    min_error = jnp.inf
    print(f"Number of parameters {num_parameters(params)}")
    print(f"Start training at epoch {init_epoch}")
    for epoch in range(init_epoch, final_epoch):
        print(f'State step on epoch {epoch}: {state_c.step}')
        rng, train_rng = jax.random.split(rng)
        state_c, metrics_update = train_fn(state_c, training_params.batch_size, train_rng) 
        if train_metrics is None:
            train_metrics = metrics_update
        else:
            train_metrics = train_metrics.merge(metrics_update)

        print(f'Epoch {epoch}: loss = {jnp.round(train_metrics.compute()["loss"], 4)}')

        is_last_step = (epoch == final_epoch - 1)

        if epoch % training_params.eval_every_steps == 0 or is_last_step:
            eval_metrics = None
            eval_state_c = eval_state_c.replace(params=state_c.params)

            with report_progress.timed('eval'):
                rollout_error_sum = 0
                for i in range(gb_c_eval._num_trajectories):
                    ts, pred_data, exp_data, aux_data, eval_metrics = rollout(eval_state_c, traj_idx=i)
                    rollout_error_sum += eval_metrics.compute()['loss']
                    plot_evaluation_curves(ts, pred_data, exp_data, aux_data,
                                           plot_dir=os.path.join(plot_dir, f'traj_{i}'),
                                           prefix=f'Epoch {epoch}: eval_traj_{i}')
                
                rollout_mean_pos_loss = rollout_error_sum / gb_c_eval._num_trajectories
                writer.write_scalars(epoch, add_prefix_to_keys({'loss': rollout_mean_pos_loss}, 'eval'))
                print(f'Epoch {epoch}: rollout mean position loss = {jnp.round(rollout_mean_pos_loss, 4)}')

                if rollout_mean_pos_loss < min_error: 
                    # Save best model
                    min_error = rollout_mean_pos_loss
                    print(f'Saving best model at epoch {epoch}')
                    with report_progress.timed('checkpoint'):
                        best_model_ckpt.save(state_c)
                if epoch > training_params.min_epochs: # train at least for 'min_epochs' epochs
                    early_stop = early_stop.update(rollout_mean_pos_loss)
                    if early_stop.should_stop:
                        print(f'Met early stopping criteria, breaking at epoch {epoch}')
                        training_params.num_epochs = epoch - init_epoch
                        is_last_step = True

        if epoch % training_params.log_every_steps == 0 or is_last_step:
            writer.write_scalars(epoch, add_prefix_to_keys(train_metrics.compute(), 'train'))
            train_metrics = None

        if epoch % training_params.ckpt_every_steps == 0 or is_last_step:
            with report_progress.timed('checkpoint'):
                ckpt.save(state_c)

        if epoch % training_params.clear_cache_every_steps == 0 or is_last_step: 
            jax.clear_caches()

        if is_last_step:
            break

    """ Evaluation """
    # rollout_error_sum = 0
    # for i in range(len(gb_c._data)):
    #     ts, pred_data, exp_data, aux_data, eval_metrics = rollout(eval_state, traj_idx=i)
    #     writer.write_scalars(
    #         i, add_prefix_to_keys(eval_metrics.compute(), f'eval {paths.evaluation_data_path_comp}')
    #     )
    #     rollout_error_sum += eval_metrics.compute()['loss']
    #     plot_evaluation_curves(ts, pred_data, exp_data, aux_data, plot_dir=eval_plot_dir, prefix=f'eval_traj_{i}')

    # print('Mean rollout error: ', rollout_error_sum / len(gb_c._data))

    # # Save evaluation metrics to json
    # eval_metrics = {
    #     'mean_rollout_error': (rollout_error_sum / len(gb_c._data)).tolist()
    # }
    # eval_metrics_file = os.path.join(plot_dir, 'eval_metrics.js')
    # with open(eval_metrics_file, "w") as outfile:
    #     json.dump(eval_metrics, outfile)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default=None)
    args = parser.parse_args()

    config = create_comp_gnn_config(args)
    train_comp(config)