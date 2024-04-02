import os
import jax
import optax

import matplotlib.pyplot as plt
import jax.numpy as jnp

import ml_collections
from clu import metric_writers
from clu import periodic_actions
from clu import checkpoint

from flax.training.train_state import TrainState
from flax.training.early_stopping import EarlyStopping

from time import strftime
from timeit import default_timer
from argparse import ArgumentParser
from graph_builder import DMSDGraphBuilder
from scripts.models import *
from utils.data_utils import *
from utils.jax_utils import *
from utils.train_utils import *
from utils.plotter import plot

def create_gnn_config(args) -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.paths = ml_collections.ConfigDict({
        'dir': args.dir,
        'training_data_path': 'results/double_mass_spring_data/train_1000_0.1_0.4.pkl',
        'evaluation_data_path': 'results/double_mass_spring_data/val_20_0.1_0.5.pkl',
    })
    config.training_params = ml_collections.ConfigDict({
        'net_name': 'GraphNet',
        'num_epochs': int(5e2),
        'min_epochs': int(0),
        'batch_size': 2,
        'rollout_timesteps': 1500,
        'log_every_steps': 1,
        'eval_every_steps': 5,
        # 'checkpoint_every_steps': 100,
        'clear_cache_every_steps': 10,
        'add_undirected_edges': True,
        'add_self_loops': True,
        'train_multi_trajectories': True,
    })
    config.optimizer_params = ml_collections.ConfigDict({
        'learning_rate': optax.exponential_decay(init_value=1e-3, transition_steps=5e2, decay_rate=0.1, end_value=1e-5),
    })
    config.net_params = ml_collections.ConfigDict({
        'prediction': 'acceleration',
        'integration_method': 'SemiImplicitEuler', 
        # 'horizon': 5, # for gnode only
        'vel_history': 5,
        'control_history': 5,
        'num_mp_steps': 1, # too big causes oversmoothing
        'noise_std': 0.0003,
        'latent_size': 16,
        'hidden_layers': 2,
        'activation': 'relu',
        'use_edge_model': True,
        'layer_norm': True,
        'shared_params': False,
        'dropout_rate': 0.5,
        'add_undirected_edges': config.training_params.add_undirected_edges,
        'add_self_loops': config.training_params.add_self_loops,
    })
    return config

def eval(config: ml_collections.ConfigDict):
    training_params = config.training_params
    net_params = config.net_params
    paths = config.paths

    def create_net():
        match training_params.net_name:
            case 'GraphNet':
                return GraphNet(**net_params)
            case 'GNODE':
                return GNODE(**net_params)
            case _:
                raise RuntimeError('Invalid net name')
            
    log_dir = os.path.join(paths.dir, 'log')
    checkpoint_dir = os.path.join(paths.dir, 'checkpoint')
    plot_dir = os.path.join(paths.dir, 'eval_plots')

    rng = jax.random.key(0)
    rng, init_rng, net_rng = jax.random.split(rng, 3)

    # Create writer for logs
    writer = metric_writers.create_default_writer(logdir=log_dir)

    tx = optax.adam(**config.optimizer_params)
    eval_gb = DMSDGraphBuilder(paths.evaluation_data_path, 
                               training_params.add_undirected_edges, 
                               training_params.add_self_loops, 
                               net_params.prediction, 
                               net_params.vel_history,
                               net_params.control_history)
    
    net_params.norm_stats = eval_gb._norm_stats
    eval_net = create_net()
    eval_net.training = False
    init_control = eval_gb._control[0, 0]
    init_graph = eval_gb.get_graph(traj_idx=0, t=net_params.vel_history+1)
    params = eval_net.init(init_rng, init_graph, init_control, net_rng)
    batched_apply = jax.vmap(eval_net.apply, in_axes=(None,0,0,None))
    state = TrainState.create(
        apply_fn=batched_apply,
        params=params,
        tx=tx,
    )
    ckpt = checkpoint.Checkpoint(checkpoint_dir)
    state = ckpt.restore_or_initialize(state)

    def rollout(eval_state: TrainState, traj_idx: int = 0, t0: int = 0):
        tf_idxs = (t0 + jnp.arange(training_params.rollout_timesteps // eval_net.num_mp_steps)) * eval_net.num_mp_steps
        t0 = round(eval_net.vel_history /  eval_net.num_mp_steps) * eval_net.num_mp_steps
        tf_idxs = jnp.unique(tf_idxs.clip(min=t0 + eval_net.num_mp_steps, max=1500))
        ts = tf_idxs * eval_net.dt

        controls = eval_gb._control[traj_idx, tf_idxs]
        exp_qs_buffer = eval_gb._qs[traj_idx, tf_idxs]
        exp_as_buffer = eval_gb._accs[traj_idx, tf_idxs]
        graphs = eval_gb.get_graph(traj_idx, t0)
        batched_graph = pytrees_stack([graphs])
        def forward_pass(graph, control):
            graph = eval_state.apply_fn(state.params, graph, jnp.array([control]), jax.random.key(0))
            pred_qs = graph.nodes[:,:,0]
            if net_params.prediction == 'acceleration':
                pred_accs = graph.nodes[:,:,-1]
                graph = graph._replace(nodes=graph.nodes[:,:,:-1]) # remove acceleration  
                return graph, (pred_qs.squeeze(), pred_accs.squeeze())
            elif net_params.prediction == 'position':
                return graph, pred_qs.squeeze()
            
        start = default_timer()
        final_batched_graph, pred_data = jax.lax.scan(forward_pass, batched_graph, controls)
        end = default_timer()
        jax.debug.print('Inference time {} [sec] for {} passes', end - start, len(ts))

        eval_pos_loss = optax.l2_loss(predictions=pred_data[0], targets=exp_qs_buffer).mean()

        if net_params.prediction == 'acceleration':
            aux_data = (eval_gb._m[traj_idx], eval_gb._k[traj_idx], eval_gb._b[traj_idx])
            return ts, np.array(pred_data), np.array((exp_qs_buffer, exp_as_buffer)), aux_data, EvalMetrics.single_from_model_output(loss=eval_pos_loss)
        elif net_params.prediction == 'position':
            return ts, np.array(pred_data), np.array(exp_qs_buffer)  

    print(f"Number of parameters {num_parameters(params)}")
    rollout_error_sum = 0
    for i in range(len(eval_gb._data)):
        ts, pred_data, exp_data, aux_data, eval_metrics = rollout(state, traj_idx=i)
        writer.write_scalars(i, add_prefix_to_keys(eval_metrics.compute(), 'eval'))
        rollout_error_sum += eval_metrics.compute()['loss']
        plot(ts, pred_data, exp_data, aux_data, plot_dir=plot_dir, prefix=f'eval_traj_{i}')

    print('Mean rollout error: ', rollout_error_sum / len(eval_gb._data))

def train(config: ml_collections.ConfigDict):
    training_params = config.training_params
    net_params = config.net_params
    paths = config.paths

    def create_net():
        match training_params.net_name:
            case 'GraphNet':
                return GraphNet(**net_params)
            case 'GNODE':
                return GNODE(**net_params)
            case _:
                raise RuntimeError('Invalid net name')
    
    if paths.dir == None:
        config.paths.dir = os.path.join(os.curdir, f'results/test_models/{strftime("%m%d")}_test_gnn/control_300_noisy_history_5_{strftime("%H%M%S")}')
        paths.dir = config.paths.dir
            
    log_dir = os.path.join(paths.dir, 'log')
    checkpoint_dir = os.path.join(paths.dir, 'checkpoint')
    plot_dir = os.path.join(paths.dir, 'plots')

    rng = jax.random.key(0)
    rng, init_rng, net_rng = jax.random.split(rng, 3)

    # Create writer for logs
    writer = metric_writers.create_default_writer(logdir=log_dir)
    # Create optimizer
    tx = optax.adam(**config.optimizer_params)
    # Create training and evaluation data loaders
    train_gb = DMSDGraphBuilder(paths.training_data_path, 
                                training_params.add_undirected_edges, 
                                training_params.add_self_loops, 
                                net_params.prediction, 
                                net_params.vel_history,
                                net_params.control_history)
    eval_gb = DMSDGraphBuilder(paths.evaluation_data_path, 
                               training_params.add_undirected_edges, 
                               training_params.add_self_loops, 
                               net_params.prediction, 
                               net_params.vel_history,
                               net_params.control_history)
    # Initialize training network
    net_params.norm_stats = train_gb._norm_stats
    net = create_net()
    init_graph = train_gb.get_graph(traj_idx=0, t=net_params.vel_history+1)
    init_control = train_gb._control[0, 0]
    params = net.init(init_rng, init_graph, init_control, net_rng)
    batched_apply = jax.vmap(net.apply, in_axes=(None, 0, 0, None))

    print(f"Number of parameters {num_parameters(params)}")
    time_offset = net.horizon if training_params.net_name == 'gnode' else net.num_mp_steps

    def random_batch(batch_size: int, min: int, max: int, rng: jax.Array):
        """ Return random permutation of jnp.arange(min, max) in batches of batch_size """
        steps_per_epoch = (max - min)// batch_size
        perms = jax.random.permutation(rng, max - min)
        perms = perms[: steps_per_epoch * batch_size].reshape(-1,batch_size)
        return perms
    
    def train_epoch(state: TrainState, batch_size: int, rng: jax.Array):
        ''' Train one epoch using all trajectories '''     
        traj_perms = random_batch(batch_size, 0, train_gb._data.shape[0], rng)
        t0_perms = random_batch(batch_size, net_params.vel_history, train_gb._data.shape[1]-time_offset, rng)
        dropout_rng = jax.random.split(rng, batch_size)
        rng, net_rng = jax.random.split(rng)

        def loss_fn(params, batch_graphs, batch_data):
            batch_targets, batch_control = batch_data
            pred_graphs = state.apply_fn(params, batch_graphs, batch_control, net_rng, rngs={'dropout': dropout_rng})
            if net_params.prediction == 'acceleration':
                predictions = pred_graphs.nodes[:,:,-1]
            elif net_params.prediction == 'position':
                predictions = pred_graphs.nodes[:,:,0]
            loss = int(1e6) * optax.l2_loss(predictions=predictions, targets=batch_targets).mean()
            return loss

        def train_batch(state, trajs, t0s):
            tfs = t0s + time_offset
            batch_accs = train_gb._accs[trajs, tfs]
            batch_control = train_gb._control[trajs, tfs]
            batch_data = (batch_accs, batch_control)
            graphs = train_gb.get_graph_batch(trajs, t0s)
            # batch_graph = pytrees_stack(graphs) # explicitly batch graphs
            loss, grads = jax.value_and_grad(loss_fn)(state.params, graphs, batch_data)
            state = state.apply_gradients(grads=grads)
            return state, loss
        
        state, epoch_loss = double_scan(train_batch, state, traj_perms, t0_perms) # TODO: switch order

        train_loss = jnp.asarray(epoch_loss).mean()

        return state, TrainMetrics.single_from_model_output(loss=train_loss)

    def train_epoch_1_traj(state: TrainState, batch_size: int, rng: jax.Array):
        ''' Train one epoch using just one trajectory (traj_idx = 0) '''     
        traj_idx = 0
        t0_perms = random_batch(batch_size, net_params.vel_history, train_gb._data.shape[1]-time_offset, rng)

        dropout_rng = jax.random.split(rng, batch_size)
        rng, net_rng = jax.random.split(rng)
        def loss_fn(params, batch_graphs, batch_data):
            batch_targets, batch_control = batch_data
            pred_graphs = state.apply_fn(params, batch_graphs, batch_control, net_rng, rngs={'dropout': dropout_rng})
            predictions = pred_graphs.nodes[:,:,-1]
            loss = int(1e6) * optax.l2_loss(predictions=predictions, targets=batch_targets).mean()
            return loss
        
        def train_batch(state, t0s):
            tfs = t0s + time_offset
            batch_accs = train_gb._accs[traj_idx, tfs]
            batch_control = train_gb._control[traj_idx, tfs]
            batch_data = (batch_accs, batch_control)
            traj_idxs = traj_idx * jnp.ones(jnp.shape(t0s), dtype=jnp.int32)
            graphs = train_gb.get_graph_batch(traj_idxs, t0s)
            loss, grads = jax.value_and_grad(loss_fn)(state.params, graphs, batch_data)

            state = state.apply_gradients(grads=grads)

            return state, loss
        
        state, epoch_loss = jax.lax.scan(train_batch, state, t0_perms)
        train_loss = jnp.asarray(epoch_loss).mean()

        return state, TrainMetrics.single_from_model_output(loss=train_loss)

    def rollout(eval_state: TrainState, traj_idx: int = 0, t0: int = 0):
        tf_idxs = (t0 + jnp.arange(training_params.rollout_timesteps // net.num_mp_steps)) * net.num_mp_steps
        t0 = round(net.vel_history /  net.num_mp_steps) * net.num_mp_steps
        tf_idxs = jnp.unique(tf_idxs.clip(min=t0 + net.num_mp_steps, max=1500))
        ts = tf_idxs * net.dt
        controls = eval_gb._control[traj_idx, tf_idxs]
        exp_qs_buffer = eval_gb._qs[traj_idx, tf_idxs]
        exp_as_buffer = eval_gb._accs[traj_idx, tf_idxs]
        graphs = eval_gb.get_graph(traj_idx, t0)
        batched_graph = pytrees_stack([graphs])
        def forward_pass(graph, control):
            graph = eval_state.apply_fn(state.params, graph, jnp.array([control]), jax.random.key(0))
            pred_qs = graph.nodes[:,:,0]
            if net_params.prediction == 'acceleration':
                pred_accs = graph.nodes[:,:,-1]
                graph = graph._replace(nodes=graph.nodes[:,:,:-1]) # remove acceleration  
                return graph, (pred_qs.squeeze(), pred_accs.squeeze())
            elif net_params.prediction == 'position':
                return graph, pred_qs.squeeze()
        final_batched_graph, pred_data = jax.lax.scan(forward_pass, batched_graph, controls)

        eval_pos_loss = optax.l2_loss(predictions=pred_data[0], targets=exp_qs_buffer).mean()

        if net_params.prediction == 'acceleration':
            aux_data = (eval_gb._m[traj_idx], eval_gb._k[traj_idx], eval_gb._b[traj_idx])
            return ts, np.array(pred_data), np.array((exp_qs_buffer, exp_as_buffer)), aux_data, EvalMetrics.single_from_model_output(loss=eval_pos_loss)
        elif net_params.prediction == 'position':
            return ts, np.array(pred_data), np.array(exp_qs_buffer)    

    state = TrainState.create(
        apply_fn=batched_apply,
        params=params,
        tx=tx,
    )

    # Create evaluation network
    eval_net = create_net()
    eval_net.training = False
    eval_net.norm_stats = eval_gb._norm_stats
    if training_params.net_name == 'GNODE': eval_net.horizon = 1 
    eval_state = state.replace(apply_fn=eval_net.apply)

    # Create logger to report training progress
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=training_params.num_epochs,
        writer=writer
    )
    # Load previous checkpoint (if applicable)
    ckpt = checkpoint.Checkpoint(checkpoint_dir)
    state = ckpt.restore_or_initialize(state)

    trajs_size = len(train_gb._data)
    ts_size = len(train_gb._data[0]) - train_gb._vel_history
    steps_per_epoch = ts_size // training_params.batch_size
    # Setup train_fn (use train_epoch or train_epoch_1_traj)
    if training_params.train_multi_trajectories:
        steps_per_epoch *= trajs_size // training_params.batch_size
        train_fn = train_epoch
    else:
        eval_gb = train_gb
        train_fn = train_epoch_1_traj
    # Setup training epochs
    init_epoch = int(state.step) // steps_per_epoch + 1
    final_epoch = init_epoch + training_params.num_epochs
    training_params.num_epochs = final_epoch

    early_stop = EarlyStopping(min_delta=1e-3, patience=2)

    train_metrics = None
    min_error = jnp.inf
    print("Start training")
    for epoch in range(init_epoch, final_epoch):
        rng, train_rng = jax.random.split(rng)
        state, metrics_update = train_fn(state, training_params.batch_size, train_rng) 
        if train_metrics is None:
            train_metrics = metrics_update
        else:
            train_metrics = train_metrics.merge(metrics_update)

        print(f'Epoch {epoch}: loss = {jnp.round(train_metrics.compute()["loss"], 4)}')

        is_last_step = (epoch == final_epoch - 1)

        if epoch % training_params.eval_every_steps == 0 or is_last_step:
            eval_metrics = None
            net.training = False
            with report_progress.timed('eval'):
                # eval_state = eval_state.replace(params=state.params)
                rollout_error_sum = 0
                for i in range(len(eval_gb._data)):
                    ts, pred_data, exp_data, aux_data, eval_metrics = rollout(state, traj_idx=i)
                    rollout_error_sum += eval_metrics.compute()['loss']
                    plot(ts, pred_data, exp_data, aux_data,
                         plot_dir=os.path.join(plot_dir, f'traj_{i}'),
                         prefix=f'Epoch {epoch}: eval_traj_{i}')
                
                rollout_mean_pos_loss = rollout_error_sum / len(eval_gb._data)
                writer.write_scalars(epoch, add_prefix_to_keys({'loss': rollout_mean_pos_loss}, 'eval'))
                print(f'Epoch {epoch}: rollout mean position loss = {jnp.round(rollout_mean_pos_loss, 4)}')

                if rollout_mean_pos_loss < min_error: 
                    # Save best model
                    min_error = rollout_mean_pos_loss
                    with report_progress.timed('checkpoint'):
                        ckpt.save(state)
                if epoch > training_params['min_epochs']: # train at least for 'min_epochs' epochs
                    early_stop = early_stop.update(rollout_mean_pos_loss)
                    if early_stop.should_stop:
                        print(f'Met early stopping criteria, breaking at epoch {epoch}')
                        training_params.num_epochs = epoch - init_epoch
                        is_last_step = True
            net.training = True

        if epoch % training_params.log_every_steps == 0 or is_last_step:
            writer.write_scalars(epoch, add_prefix_to_keys(train_metrics.compute(), 'train'))
            train_metrics = None

        # if epoch % training_params.checkpoint_every_steps == 0 or is_last_step:
        #     with report_progress.timed('checkpoint'):
        #         ckpt.save(state)

        if epoch % training_params.clear_cache_every_steps == 0 or is_last_step: 
            jax.clear_caches()

        if is_last_step:
            break

    # Save config to json
    config_js = config.to_json_best_effort()
    run_params_file = os.path.join(paths.dir, 'run_params.js')
    with open(run_params_file, "w") as outfile:
        json.dump(config_js, outfile)


def test_graph_net(config: ml_collections.ConfigDict):
    training_params = config.training_params
    net_params = config.net_params
    rng = jax.random.key(0)
    rng, init_rng = jax.random.split(rng)
    batch_size = 1
    tx = optax.adam(1e-3)
    gb = DMSDGraphBuilder(config.config.training_data_path, 
                          training_params.add_undirected_edges,
                          training_params.add_self_loops, 
                          training_params.vel_history)
    net_params.normalization_stats = gb._norm_stats

    net = GraphNet(**net_params)
    init_graph = gb.get_graph_batch(jnp.zeros(training_params.batch_size), 
                                    jnp.ones(training_params.batch_size, dtype=jnp.int32) * training_params.vel_history + 1)
    params = net.init(init_rng, init_graph)
    batched_apply = jax.vmap(net.apply, in_axes=(None,0))

    state = TrainState.create(
        apply_fn=batched_apply,
        params=params,
        tx=tx,
    )

    batched_graph = pytrees_stack(init_graph)
    y = jnp.ones((batch_size, gb.n_node, training_params.vel_history+1)) # [batch_size, graph nodes, graph features]
    def loss_fn(param, graph, targets):
        pred_graph = state.apply_fn(param, graph)
        pred_nodes = pred_graph.nodes
        loss = optax.l2_loss(predictions=pred_nodes, targets=targets).mean()
        return loss
    
    print(loss_fn(state.params, batched_graph, y))
    
    grads = jax.grad(loss_fn)(state.params, batched_graph, y)
    state = state.apply_gradients(grads=grads)

    print(loss_fn(state.params, batched_graph, y))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    config = create_gnn_config(args)

    if args.eval:
        eval(config)
    else:
        train(config)

    # For testing:
    # test_graph_net(config)