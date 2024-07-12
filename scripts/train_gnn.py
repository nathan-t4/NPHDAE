import os
import jax
import optax
import json

import jax.numpy as jnp

import ml_collections
from clu import metric_writers
from clu import periodic_actions

from flax.training.train_state import TrainState
from flax.training.early_stopping import EarlyStopping

import orbax.checkpoint as ocp

from timeit import default_timer
from argparse import ArgumentParser
from scripts.models import *
from scripts.graph_nets import *
from utils.train_utils import *
from utils.data_utils import *
from utils.jax_utils import *
from utils.gnn_utils import *

from helpers.graph_builder_factory import gb_factory
from helpers.config_factory import config_factory

# from absl import logging
# logging.set_verbosity(logging.INFO)

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.5'

def eval(config: ml_collections.ConfigDict):
    # TODO: fix T!
    training_params = config.training_params
    net_params = config.net_params
    paths = config.paths
    
    name = set_name(config)
    log_dir = os.path.join(paths.dir, 'log')
    checkpoint_dir = os.path.join(paths.dir, 'checkpoint')
    plot_dir = os.path.join(paths.dir, 'eval_plots')

    rng = jax.random.key(config.seed)
    rng, init_rng, net_rng = jax.random.split(rng, 3)

    # Create writer for logs
    writer = metric_writers.create_default_writer(logdir=log_dir)

    tx = optax.adam(**config.optimizer_params)

    graph_builder = gb_factory(name)

    train_gb = graph_builder(paths.training_data_path)
    eval_gb = graph_builder(paths.evaluation_data_path)
    
    t0 = 0
    init_control = eval_gb._control[0, 0]

    net_params.training = False
    net_params.graph_from_state = train_gb.get_graph_from_state
    net_params.include_idxs = None
    net_params.edge_idxs = train_gb.edge_idxs
    net_params.node_idxs = train_gb.node_idxs

    if not training_params.learn_matrices:
        net_params.J = train_gb.J
        net_params.R = train_gb.R
        net_params.g = train_gb.g

    eval_net = create_net(name, net_params)
    eval_net.training = False
    time_offset = eval_net.T
    init_graph = eval_gb.get_graph(traj_idx=0, t=t0)
    params = eval_net.init(init_rng, init_graph, init_control, net_rng)
    state = TrainState.create(
        apply_fn=eval_net.apply,
        params=params,
        tx=tx,
    )
    checkpoint_dir = os.path.join(checkpoint_dir, 'best_model')
    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)  
    checkpoint_dir = os.path.abspath(checkpoint_dir)  
    ckpt_mngr = ocp.CheckpointManager(
        checkpoint_dir,
        options=options,
        item_handlers=ocp.StandardCheckpointHandler(),
    )
    state = ckpt_mngr.restore(paths.ckpt_step, args=ocp.args.StandardRestore(state))

    def rollout(eval_state: TrainState, traj_idx: int, ti: int = 0):
        tf_idxs = ti + jnp.arange(1, jnp.floor_divide(training_params.rollout_timesteps + 1, eval_net.T))
        tf_idxs = jnp.unique(tf_idxs.clip(min=ti + 1, max=jnp.floor_divide(eval_gb._num_timesteps + 1, eval_net.T))) * eval_net.T
        t0_idxs = tf_idxs - time_offset
        ts = tf_idxs * eval_net.dt
        graph = eval_gb.get_graph(traj_idx, ti)
        controls = eval_gb.get_control(traj_idx, t0_idxs)
        exp_data = eval_gb.get_exp_data(traj_idx, tf_idxs)
        get_pred_data = eval_gb.get_pred_data

        if name == 'MassSpring':
            controls = eval_gb._control[traj_idx, t0_idxs]
            exp_qs_buffer = eval_gb._qs[traj_idx, tf_idxs]
            exp_as_buffer = eval_gb._accs[traj_idx, tf_idxs]
        
        def forward_pass(graph, control):
            if name == 'MassSpring':
                graph = eval_state.apply_fn(eval_state.params, graph, control, jax.random.key(config.seed))
                pred_qs = (graph.nodes[:,0]).squeeze()
                pred_accs = (graph.nodes[:,-1]).squeeze()
                graph = graph._replace(nodes=graph.nodes[:,:-1]) # remove acceleration  
                return graph, (pred_qs, pred_accs)
            
            graph = eval_state.apply_fn(eval_state.params, graph, control, jax.random.key(config.seed))
            pred_data = get_pred_data(graph)
            graph = graph._replace(globals=None)
            return graph, pred_data

        start = default_timer()
        final_batched_graph, pred_data = jax.lax.scan(forward_pass, graph, controls)
        end = default_timer()
        jax.debug.print('Inference time {} [sec] for {} passes', end - start, len(ts))
        if name == 'MassSpring':
            rollout_loss = optax.l2_loss(predictions=pred_data[0], targets=exp_qs_buffer).mean()
            aux_data = {
                'name': name,
                'm': eval_gb._m[traj_idx], 
                'k': eval_gb._k[traj_idx], 
                'b': eval_gb._b[traj_idx],
            }
            return ts, np.array(pred_data), np.array((exp_qs_buffer, exp_as_buffer)), aux_data, EvalMetrics.single_from_model_output(loss=rollout_loss)
        elif 'LC' in name:
            losses = [jnp.sum(optax.l2_loss(predictions=pred_data[i], targets=exp_data[i])) for i in range(len(exp_data))]
            eval_metrics = [EvalMetrics.single_from_model_output(loss=loss) for loss in losses]
            aux_data = {
                'name': name,
            }
            return ts, np.array(pred_data), np.array(exp_data), aux_data, eval_metrics
        
        else:
            raise NotImplementedError() 
        
    trajs_size = train_gb._num_timesteps
    ts_size = train_gb._num_timesteps - t0
    steps_per_epoch = (ts_size * trajs_size) // (training_params.batch_size ** 2)
    init_epoch = int(state.step) // steps_per_epoch + 1
    print(f"Number of parameters {num_parameters(params)}")
    print(f"Number of epochs {init_epoch}")
    rollout_error_sum = 0
    error_sums = [0] * eval_gb._num_states
    for i in range(eval_gb._num_trajectories):
        if name == 'MassSpring':
            ts, pred_data, exp_data, aux_data, eval_metrics = rollout(state, traj_idx=i)
            rollout_error_sum += eval_metrics.compute()['loss']   
        elif 'LC' in name:
            ts, pred_data, exp_data, aux_data, eval_metrics = rollout(state, traj_idx=i)
            for j in range(len(error_sums)):
                error_sums[j] += eval_metrics[j].compute()['loss']
        else:
            raise NotImplementedError()
        
        error_sums = np.array(error_sums)
        
        plot_evaluation_curves(ts, pred_data, exp_data, aux_data,
                                plot_dir=plot_dir,
                                prefix=f'eval_traj_{i}')
        writer.write_scalars(i, add_prefix_to_keys({'loss': error_sums.mean() / eval_gb._num_timesteps}, 'eval'))
        
    rollout_error = np.Inf
    if name == 'MassSpring':
        rollout_error = rollout_error_sum / eval_gb._num_trajectories
        print(f'Rollout mean position loss = {jnp.round(rollout_error, 4)}')
        
    elif 'LC' in name:
        rollout_error_state = error_sums / (eval_gb._num_trajectories * eval_gb._num_timesteps)
        rollout_error = rollout_error_state.mean()
        print(f'State errors {rollout_error_state}')
        print(f'Rollout error {rollout_error}')

    # Save evaluation metrics to json
    eval_metrics = {
        'rollout_error_state': rollout_error_state.tolist(),
        'rollout_error': str(rollout_error),
        'evaluation_data_path': paths.evaluation_data_path,
    }
    eval_metrics_file = os.path.join(plot_dir, 'eval_metrics.js')
    with open(eval_metrics_file, "w") as outfile:
        json.dump(eval_metrics, outfile)

def train(config: ml_collections.ConfigDict, optuna_trial = None):
    training_params = config.training_params
    net_params = config.net_params
    paths = config.paths

    name = set_name(config)
    
    restore = True if paths.dir else False

    if paths.dir is None:
        config.paths.dir = os.path.join(
            os.curdir, 
            f'results/{training_params.net_name}/{config.system_name}/{config.trial_name}')
        paths.dir = config.paths.dir
    
    log_dir = os.path.join(paths.dir, 'log')
    plot_dir = os.path.join(paths.dir, 'plots')
    checkpoint_dir = os.path.join(paths.dir, 'checkpoint')
    checkpoint_dir = os.path.join(checkpoint_dir, 'best_model')
    checkpoint_dir = os.path.abspath(checkpoint_dir)

    rng = jax.random.key(config.seed)
    rng, init_rng, net_rng = jax.random.split(rng, 3)

    # Create writer for logs
    writer = metric_writers.create_default_writer(logdir=log_dir)
    # Create optimizer
    tx = optax.adam(**config.optimizer_params)
    # Create training and evaluation data loaders
    graph_builder = gb_factory(name)
    train_gb = graph_builder(paths.training_data_path)
    eval_gb = graph_builder(paths.evaluation_data_path)
    net_params.training = True
    net_params.graph_from_state = train_gb.get_graph_from_state
    net_params.include_idxs = None
    net_params.edge_idxs = train_gb.edge_idxs
    net_params.node_idxs = train_gb.node_idxs
    t0 = 0

    # if not training_params.learn_matrices:
    #     net_params.J = train_gb.J
    #     net_params.R = train_gb.R
    #     net_params.g = train_gb.g

    if name == 'MassSpring':
        net_params.norm_stats = train_gb._norm_stats
        t0 = net_params.vel_history
        init_control = train_gb._control[0, t0, :]

    init_control = train_gb.get_control(0, t0)
    # Initialize training network
    init_graph = train_gb.get_graph(0, t0)
    net = create_net(name, net_params)
    params = net.init(init_rng, init_graph, init_control, net_rng)
    batched_apply = jax.vmap(net.apply, in_axes=(None, 0, 0, None))

    # logging.log(logging.INFO, f"Number of parameters {num_parameters(params)}")
    print(f"Number of parameters {num_parameters(params)}")
    time_offset = net.T

    def random_batch(batch_size: int, min: int, max: int, rng: jax.Array):
        """ Return random permutation of jnp.arange(min, max) in batches of batch_size """
        steps_per_epoch = (max - min)// batch_size
        perms = jax.random.permutation(rng, max - min)
        perms = perms[: steps_per_epoch * batch_size].reshape(-1,batch_size)
        return perms
    
    def train_epoch(state: TrainState, batch_size: int, rng: jax.Array):
        ''' Train one epoch using all trajectories '''     
        loss_function = training_params.loss_function

        traj_perms = random_batch(batch_size, 0, train_gb._num_trajectories, rng)
        t0_perms = random_batch(batch_size, t0, train_gb._num_timesteps-time_offset, rng) 
        dropout_rng = jax.random.split(rng, batch_size)
        rng, net_rng = jax.random.split(rng)

        def loss_fn(params, batch_graphs, batch_data):
            if name == 'MassSpring' and loss_function == 'acceleration':
                batch_targets, batch_control = batch_data
                pred_graphs = state.apply_fn(params, batch_graphs, batch_control, net_rng, rngs={'dropout': dropout_rng})
                predictions = pred_graphs.nodes[:,:,-1] 
                loss = int(1e6) * optax.l2_loss(predictions=predictions, targets=batch_targets).mean()
            elif name == 'MassSpring' and loss_function == 'state':
                batch_pos, batch_vel, batch_control = batch_data
                pred_graphs = state.apply_fn(params, batch_graphs, batch_control, net_rng, rngs={'dropout': dropout_rng})
                pos_predictions = pred_graphs.nodes[:,:,0]
                vel_predictions = pred_graphs.nodes[:,:,t0]
                loss = int(1e6) * (optax.l2_loss(predictions=pos_predictions, targets=batch_pos).mean() \
                     + optax.l2_loss(predictions=vel_predictions, targets=batch_vel).mean())
            elif name == 'LC' and loss_function == 'state':
                traj_idx = jnp.array(batch_data[0])
                Q = jnp.array(batch_data[1]).reshape(1,-1)
                Phi = jnp.array(batch_data[2]).reshape(1,-1)
                pred_graphs = state.apply_fn(params, traj_idx, batch_graphs, None, net_rng, rngs={'dropout': dropout_rng})
                predictions = pred_graphs.edges.squeeze()
                targets = jnp.concatenate((Q, Phi)).squeeze() 
                loss = optax.l2_loss(predictions, targets)
                loss = jnp.sum(loss)
            elif name == 'LC1' and loss_function == 'state':
                batch_control = jnp.array(batch_data[0])
                Q1 = jnp.array(batch_data[1]).reshape(-1,1)
                Phi1 = jnp.array(batch_data[2]).reshape(-1,1)
                Q3 = jnp.array(batch_data[3]).reshape(-1,1)
                pred_graphs = state.apply_fn(params, batch_graphs, batch_control, net_rng, rngs={'dropout': dropout_rng})
                predictions_e = pred_graphs.edges[:,:,0].squeeze()
                targets_e = jnp.concatenate((Q1, Phi1, Q3), axis=1).squeeze()
                loss = optax.squared_error(predictions_e, targets_e).mean() # MSE
            elif name == 'LC2' and loss_function == 'state':
                batch_control = jnp.array(batch_data[0])
                Q = jnp.array(batch_data[1]).reshape(-1,1)
                Phi = jnp.array(batch_data[2]).reshape(-1,1)
                pred_graphs = state.apply_fn(params, batch_graphs, batch_control, net_rng, rngs={'dropout': dropout_rng})
                predictions_e = pred_graphs.edges[:,:,0].squeeze()
                targets_e = jnp.concatenate((Q, Phi),axis=1).squeeze() # order is the same as LC2 graph edges
                loss = optax.squared_error(predictions_e, targets_e).mean() # MSE
            elif name == 'CoupledLC' and loss_function == 'state':
                batch_control = jnp.array(batch_data[0])
                Q1 = jnp.array(batch_data[1]).reshape(-1,1)
                Phi1 = jnp.array(batch_data[2]).reshape(-1,1)
                Q3 = jnp.array(batch_data[3]).reshape(-1,1)
                Q2 = jnp.array(batch_data[4]).reshape(-1,1)
                Phi2 = jnp.array(batch_data[5]).reshape(-1,1)
                pred_graphs = state.apply_fn(params, batch_graphs, batch_control, net_rng, rngs={'dropout': dropout_rng})
                predictions_e = pred_graphs.edges[:,:,0].squeeze()
                targets_e = jnp.concatenate((Q1, Phi1, Q3, Q2, Phi2),axis=1).squeeze()
                loss = optax.squared_error(predictions_e, targets_e).mean()
            elif name == 'Alternator' and loss_function == 'state':
                batch_control = jnp.array(batch_data[0])
                pred_graphs = state.apply_fn(params, batch_graphs, batch_control, net_rng, rngs={'dropout': dropout_rng})
                predictions_e = pred_graphs.edges[:,:,0].squeeze()
                targets_e = [data.reshape(-1, 1) for data in batch_data[1:-1]] # excluding control and H
                targets_e = jnp.concatenate(targets_e, axis=1).squeeze()
                loss = optax.squared_error(predictions_e, targets_e).mean()
            return loss

        def train_batch(state, trajs, t0s):
            tfs = t0s + time_offset
            batch_control = train_gb.get_control(trajs, t0s)
            batch_exp_data = train_gb.get_exp_data(trajs, tfs)
            batch_data = (batch_control, *batch_exp_data)            

            if name == 'MassSpring' and loss_function == 'acceleration':
                batch_accs = train_gb._accs[trajs, tfs]
                batch_data = (batch_accs, batch_control)
            elif name == 'MassSpring' and loss_function == 'state':
                batch_pos = train_gb._qs[trajs, tfs]
                batch_vel = train_gb._vs[trajs, tfs]
                batch_data = (batch_pos, batch_vel, batch_control)

            graphs = train_gb.get_graph_batch(trajs, t0s)
            # states = train_gb.get_state_batch(trajs, t0s) # TODO
            loss, grads = jax.value_and_grad(loss_fn)(state.params, graphs, batch_data) # TODO: replace graphs with states
            state = state.apply_gradients(grads=grads)
            return state, loss
        
        state, epoch_loss = double_scan(train_batch, state, traj_perms, t0_perms)

        train_loss = jnp.asarray(epoch_loss).mean()

        return state, TrainMetrics.single_from_model_output(loss=train_loss)

    def rollout(eval_state: TrainState, traj_idx: int, ti: int = 0):
        tf_idxs = ti + jnp.arange(1, jnp.floor_divide(training_params.rollout_timesteps + 1, net.T))
        tf_idxs = jnp.unique(tf_idxs.clip(min=ti + 1, max=jnp.floor_divide(eval_gb._num_timesteps + 1, net.T))) * net.T
        t0_idxs = tf_idxs - time_offset
        ts = tf_idxs * net.dt
        graph = eval_gb.get_graph(traj_idx, ti)
        controls = eval_gb.get_control(traj_idx, t0_idxs)
        exp_data = eval_gb.get_exp_data(traj_idx, tf_idxs)
        get_pred_data = eval_gb.get_pred_data

        if name == 'MassSpring':
            controls = eval_gb._control[traj_idx, t0_idxs]
            exp_qs_buffer = eval_gb._qs[traj_idx, tf_idxs]
            exp_as_buffer = eval_gb._accs[traj_idx, tf_idxs]
        
        def forward_pass(graph, control):
            if name == 'MassSpring':
                graph = eval_state.apply_fn(eval_state.params, graph, control, jax.random.key(config.seed))
                pred_qs = (graph.nodes[:,0]).squeeze()
                pred_accs = (graph.nodes[:,-1]).squeeze()
                graph = graph._replace(nodes=graph.nodes[:,:-1]) # remove acceleration  
                return graph, (pred_qs, pred_accs)
            
            graph = eval_state.apply_fn(eval_state.params, graph, control, jax.random.key(config.seed))
            pred_data = get_pred_data(graph)
            graph = graph._replace(globals=None)
            return graph, pred_data

        final_batched_graph, pred_data = jax.lax.scan(forward_pass, graph, controls)
        
        if name == 'MassSpring':
            rollout_loss = optax.l2_loss(predictions=pred_data[0], targets=exp_qs_buffer).mean()
            aux_data = {
                'name': name,
                'm': eval_gb._m[traj_idx], 
                'k': eval_gb._k[traj_idx], 
                'b': eval_gb._b[traj_idx],
            }
            return ts, np.array(pred_data), np.array((exp_qs_buffer, exp_as_buffer)), aux_data, EvalMetrics.single_from_model_output(loss=rollout_loss)
        elif 'LC' in name or 'Alternator' in name:
            losses = [jnp.sum(optax.l2_loss(predictions=pred_data[i], targets=exp_data[i])) for i in range(len(exp_data))]
            eval_metrics = [EvalMetrics.single_from_model_output(loss=loss) for loss in losses]
            aux_data = {
                'name': name,
            }
            return ts, np.array(pred_data), np.array(exp_data), aux_data, eval_metrics
        
        else:
            raise NotImplementedError()

    state = TrainState.create(
        apply_fn=batched_apply,
        params=params,
        tx=tx,
    )

    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)        
    ckpt_mngr = ocp.CheckpointManager(
        checkpoint_dir,
        options=options,
        item_handlers=ocp.StandardCheckpointHandler(),
    )

    # Restore state from checkpoint
    if restore: state = ckpt_mngr.restore(paths.ckpt_step, args=ocp.args.StandardRestore(state))

    # Create evaluation network
    eval_net = create_net(name, net_params)
    eval_net.training = False
    # Use same normalization stats as training set
    if 'mass_spring' in name:
        eval_net.norm_stats = train_gb._norm_stats  # for mass_spring
    eval_state = state.replace(apply_fn=eval_net.apply)

    # Create logger to report training progress
    report_progress = periodic_actions.ReportProgress(
            num_train_steps=training_params.num_epochs,
            writer=writer
        )
    profiler = periodic_actions.Profile(logdir=log_dir)
    hooks = [report_progress, profiler]
    ts_size = (train_gb._num_timesteps - t0)
    trajs_size = train_gb._num_trajectories
    steps_per_epoch = int((ts_size * trajs_size) // (training_params.batch_size ** 2))

    # Setup training epochs
    init_epoch = state.step // steps_per_epoch + 1
    final_epoch = init_epoch + training_params.num_epochs
    training_params.num_epochs = final_epoch

    early_stop = EarlyStopping(min_delta=1e-3, patience=2)

    train_metrics = None
    min_error = jnp.inf
    ckpt_step = init_epoch
    # logging.info(f"Start training at epoch {init_epoch}")
    print(f"Start training at epoch {init_epoch}")
    for epoch in range(init_epoch, final_epoch):
        rng, train_rng = jax.random.split(rng)
        # with jax.profiler.StepTraceAnnotation('train', step_num=epoch):
        state, metrics_update = train_epoch(state, training_params.batch_size, train_rng) 
        if train_metrics is None:
            train_metrics = metrics_update
        else:
            train_metrics = train_metrics.merge(metrics_update)

        print(f'Epoch {epoch}: ' + 'loss = {:.15}'.format(train_metrics.compute()["loss"]))

        # for hook in hooks:
        #     hook(epoch)

        is_last_step = (epoch == final_epoch - 1)

        if epoch % config.eval_every_steps == 0 or is_last_step:
            eval_metrics = None
            eval_state = eval_state.replace(params=state.params)

            with report_progress.timed('eval'):
                rollout_error_sum = 0
                error_sums = [0] * train_gb._num_states
                num_eval_trajs = 2 # eval_gb._num_trajectories
                for i in range(num_eval_trajs):
                    if name == 'MassSpring':
                        ts, pred_data, exp_data, aux_data, eval_metrics = rollout(eval_state, traj_idx=i)
                        rollout_error_sum += eval_metrics.compute()['loss']   
                    elif 'LC' in name or name == 'Alternator':
                        ts, pred_data, exp_data, aux_data, eval_metrics = rollout(eval_state, traj_idx=i)
                        for j in range(len(error_sums)):
                            error_sums[j] += eval_metrics[j].compute()['loss']
                    else:
                        raise NotImplementedError()
                    
                    plot_evaluation_curves(ts, pred_data, exp_data, aux_data,
                                           plot_dir=os.path.join(plot_dir, f'traj_{i}'),
                                           prefix=f'Epoch {epoch}: eval_traj_{i}')
                rollout_error = np.Inf
                if name == 'MassSpring':
                    rollout_error = rollout_error_sum / eval_gb._num_trajectories
                    # print(f'Epoch {epoch}: rollout mean position loss = {jnp.round(rollout_error, 4)}')
                    
                else:
                    rollout_error_state = np.array(error_sums) / (num_eval_trajs * eval_gb._num_timesteps) # TODO: was eval_gb._num_trajectories
                    print(f'Epoch {epoch} state errors {rollout_error_state}')
                    rollout_error = rollout_error_state.mean()

                writer.write_scalars(epoch, add_prefix_to_keys({'loss': rollout_error}, 'eval'))

                if rollout_error < min_error: 
                    # Save best model
                    min_error = rollout_error
                    ckpt_step = epoch
                    # logging.info(f'Saving best model at epoch {epoch}')
                    print(f'Saving best model at epoch {epoch}')
                    with report_progress.timed('checkpoint'):
                        ckpt_mngr.save(epoch, args=ocp.args.StandardSave(state))
                        ckpt_mngr.wait_until_finished()
                    
                    if net.J is None: # if training J
                        J_triu = jnp.triu(state.params['params']['J']['kernel'])
                        J = J_triu - J_triu.T
                        # logging.info(f'upper triangular of J: {J}')
                        print(f'upper triangular of J: {J}')

                    if net.g is None: # if training g
                        pass # TODO

                if epoch > training_params.min_epochs: # train for at least 'min_epochs' epochs
                    early_stop = early_stop.update(rollout_error)
                    if early_stop.should_stop:
                        print(f'Met early stopping criteria, breaking at epoch {epoch}')
                        training_params.num_epochs = epoch - init_epoch
                        is_last_step = True
                if optuna_trial:
                    optuna_trial.report(rollout_error, step=epoch)

        if epoch % config.log_every_steps == 0 or is_last_step:
            writer.write_scalars(epoch, add_prefix_to_keys(train_metrics.compute(), 'train'))
            train_metrics = None

        if epoch % config.clear_cache_every_steps == 0 or is_last_step: 
            jax.clear_caches()

        if is_last_step:
            break
    
    # Save training details to json
    config.metrics = ml_collections.ConfigDict()
    config.metrics.min_error = min_error.item()
    config.paths.ckpt_step = ckpt_step

    if training_params.learn_matrices:
        J_triu = jnp.triu(state.params['params']['J']['kernel'])
        J = J_triu - J_triu.T
        config.net_params.J = J.tolist()
    
    config_js = config.to_json_best_effort()
    run_params_file = os.path.join(paths.dir, 'run_params.js')
    with open(run_params_file, "w") as outfile:
        json.dump(config_js, outfile, indent=4)

    return config if not optuna_trial else config.metrics.min_error

def test_graph_net(config: ml_collections.ConfigDict):
    """ For testing """
    training_params = config.training_params
    net_params = config.net_params
    rng = jax.random.key(0)
    rng, init_rng = jax.random.split(rng)
    batch_size = 1
    tx = optax.adam(1e-3)
    gb = MSDGraphBuilder(config.config.training_data_path, 
                         training_params.add_undirected_edges,
                         training_params.add_self_loops, 
                         training_params.vel_history)
    net_params.normalization_stats = gb._norm_stats

    net = GraphNetworkSimulator(**net_params)
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
    parser.add_argument('--ckpt_step', type=int, default=None)
    parser.add_argument('--system', type=str, required=True)
    args = parser.parse_args()
    
    config = config_factory(args.system, args)

    if args.eval:
        eval(config)
    else:
        train(config)

    