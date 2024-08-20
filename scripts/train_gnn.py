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
from scripts.models.gnn import *
from scripts.model_instances.ph_gns import *
from utils.train_utils import *
from utils.data_utils import *
from utils.jax_utils import *
from utils.gnn_utils import *

from helpers.graph_builder_factory import gb_factory
from helpers.config_factory import config_factory

# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.5'

def eval(config: ml_collections.ConfigDict):
    training_params = config.training_params
    net_params = config.net_params
    paths = config.paths
    
    name = set_name(config)
    dirs = setup_dirs(config)
    dirs['log'] = os.path.join(paths.dir, 'log')
    dirs['ckpt'] = os.path.join(paths.dir, 'checkpoint')
    dirs['plot'] = os.path.join(paths.dir, 'eval_plots')

    rng = jax.random.key(config.seed)
    rng, init_rng, net_rng = jax.random.split(rng, 3)

    # Create writer for logs
    writer = metric_writers.create_default_writer(logdir=dirs['log'])

    tx = optax.adam(**config.optimizer_params)

    # Incidence matrices of the system
    AC = config.AC
    AR = config.AR
    AL = config.AL
    AV = config.AV
    AI = config.AI
    incidence_matrices = (AC, AR, AL, AV, AI)

    graph_builder = gb_factory(name)
    train_gb = graph_builder(paths.training_data_path, *incidence_matrices)
    eval_gb = graph_builder(paths.evaluation_data_path, *incidence_matrices)
    
    t0 = 0
    init_control = eval_gb._control[0, 0]

    net_params.training = False
    net_params.system_config = get_system_config(*incidence_matrices) 
    net_params.state_to_graph = train_gb.state_to_graph
    net_params.graph_to_state = train_gb.graph_to_state
    net_params.alg_vars_from_graph = train_gb.get_alg_vars_from_graph
    net_params.include_idxs = train_gb.include_idxs
    net_params.edge_idxs = train_gb.edge_idxs
    net_params.node_idxs = train_gb.node_idxs
    net_params.differential_vars = train_gb.differential_vars
    net_params.algebraic_vars = train_gb.algebraic_vars

    net = create_net(net_params)
    net.training = False
    init_graph = eval_gb.get_graph(traj_idx=0, t=t0)
    params = net.init(init_rng, init_graph, init_control, 0.0, net_rng)
    state = TrainState.create(
        apply_fn=net.apply,
        params=params,
        tx=tx,
    )
    dirs['ckpt'] = os.path.join(dirs['ckpt'], 'best_model')
    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)  
    dirs['ckpt'] = os.path.abspath(dirs['ckpt'])  
    ckpt_mngr = ocp.CheckpointManager(
        dirs['ckpt'],
        options=options,
        item_handlers=ocp.StandardCheckpointHandler(),
    )
    state = ckpt_mngr.restore(paths.ckpt_step, args=ocp.args.StandardRestore(state))

    def rollout(eval_state: TrainState, traj_idx: int, ti: int = 0):
        tf = jnp.floor_divide(training_params.rollout_timesteps + 1, net.T)
        tf_idxs = ti + jnp.arange(1, tf)
        tf_idxs = tf_idxs * net.T
        t0_idxs = tf_idxs - net_params.T
        ts = tf_idxs * net.dt
        graph = eval_gb.get_graph(traj_idx, ti)
        controls = eval_gb.get_control(traj_idx, t0_idxs)
        exp_data = eval_gb.get_exp_data(traj_idx, tf_idxs)
        get_pred_data = eval_gb.get_pred_data
        
        def forward_pass(graph, inputs):  
            control, t = inputs         
            graph = eval_state.apply_fn(eval_state.params, graph, control, t, jax.random.key(config.seed))
            pred_data = get_pred_data(graph)
            graph = graph._replace(globals=None)
            return graph, pred_data

        start = default_timer()
        _, pred_data = jax.lax.scan(forward_pass, graph, (controls, t0_idxs * net.dt))
        end = default_timer()
        
        print(f'Inference time {end-start} [sec] for {len(ts)} forward passes')

        losses = [
            jnp.sum(optax.l2_loss(predictions=pred_data[i], targets=exp_data[i])) for i in range(len(exp_data))
        ]
        eval_metrics = [EvalMetrics.single_from_model_output(loss=loss) for loss in losses]
        pred_data = np.concatenate(pred_data, axis=1)
        exp_data = np.concatenate(exp_data, axis=1)
        return pred_data, exp_data, eval_metrics


    print('##################################################')
    print(f"Start evaluation of trained model")
    print(f"\tTrained model directory: {os.path.relpath(dirs['ckpt'])}")
    print(f"\tNumber of parameters: {num_parameters(params)}")
    print(f"\tModel was trained for {config.paths.ckpt_step} epochs")
    print('##################################################')

    num_eval_trajs = eval_gb._num_trajectories
    num_eval_timesteps = training_params.rollout_timesteps
    error_sums = [0] * 4
    for i in range(num_eval_trajs):
        pred_data, exp_data, eval_metrics = rollout(state, traj_idx=i)
        for j in range(len(error_sums)):
            error_sums[j] += eval_metrics[j].compute()['loss']
        
        error_sums = np.array(error_sums)
               
        eval_gb.plot(pred_data, exp_data, 
                     plot_dir=dirs['plot'],
                     prefix=f'eval_traj_{i}')
        
        writer.write_scalars(i, add_prefix_to_keys({'loss': error_sums.mean() / num_eval_timesteps}, 'eval'))
        
    rollout_error_state = error_sums / (num_eval_trajs * num_eval_timesteps)
    rollout_error = rollout_error_state.mean()

    print('##################################################')
    print(f"Finished evaluation")
    print(f"State errors {rollout_error_state}")
    print(f"Rollout error {rollout_error}")
    print('##################################################')

    # Save evaluation metrics to json
    eval_metrics = {
        'rollout_error_state': rollout_error_state.tolist(),
        'rollout_error': str(rollout_error),
        'evaluation_data_path': paths.evaluation_data_path,
    }
    eval_metrics_file = os.path.join(dirs['plot'], 'eval_metrics.js')
    with open(eval_metrics_file, "w") as outfile:
        json.dump(eval_metrics, outfile)

def train(config: ml_collections.ConfigDict, optimizing_hparams=False):
    training_params = config.training_params
    net_params = config.net_params
    paths = config.paths

    name = config.system_name
    restore = True if paths.dir else False
    dirs = setup_dirs(config)

    rng = jax.random.key(config.seed)
    rng, init_rng, net_rng = jax.random.split(rng, 3)

    # Create writer for logs
    writer = metric_writers.create_default_writer(logdir=dirs['log'])
    tx = optax.adam(**config.optimizer_params)
    graph_builder = gb_factory(name)

    # Incidence matrices of the system
    AC = config.AC
    AR = config.AR
    AL = config.AL
    AV = config.AV
    AI = config.AI
    # Initialize graph builders - these transform the training data into jraph.GraphsTuples
    train_gb = graph_builder(paths.training_data_path, AC, AR, AL, AV, AI)
    eval_gb = graph_builder(paths.evaluation_data_path, AC, AR, AL, AV, AI)

    ############################################################
    ### Update net_params 
    net_params.training = True
    net_params.state_to_graph = train_gb.state_to_graph
    net_params.graph_to_state = train_gb.graph_to_state
    net_params.alg_vars_from_graph = train_gb.get_alg_vars_from_graph
    net_params.include_idxs = train_gb.include_idxs
    net_params.edge_idxs = train_gb.edge_idxs
    net_params.node_idxs = train_gb.node_idxs
    ############################################################
    
    ############################################################
    ### Initialize network
    init_control = train_gb.get_control(0, 0)
    # Initialize training network
    init_graph = train_gb.get_graph(0, 0)
    # Remember to check incidence matrices before training!
    incidence_matrices = (AC, AR, AL, AV, AI)
    net_params.system_config = get_system_config(*incidence_matrices) 

    print('##################################################')
    print(f'Include idxs: {net_params.include_idxs}')
    print('Incidence matrices')
    print(f"AC.T: {net_params.system_config['AC'].T}")
    print(f"AR.T: {net_params.system_config['AR'].T}")
    print(f"AL.T: {net_params.system_config['AL'].T}")
    print(f"AV.T: {net_params.system_config['AV'].T}")
    print(f"AI.T: {net_params.system_config['AI'].T}")
    print('PH matrices')
    print(f"E: {net_params.system_config['E']}")
    print(f"J: {net_params.system_config['J']}")
    print(f"B: {net_params.system_config['B']}")
    print('system indices')
    print(f"Differential indices {net_params.system_config['diff_indices']}")
    print(f"Algebraic indices {net_params.system_config['alg_indices']}")
    print('##################################################')

    t0 = 0.0
    net = create_net(net_params)
    params = net.init(init_rng, init_graph, init_control, t0, net_rng)
    batched_apply = jax.vmap(net.apply, in_axes=(None, 0, 0, 0, None))
    ############################################################
        
    def train_epoch(state: TrainState, batch_size: int, rng: jax.Array):
        ''' Train one epoch using all trajectories '''     
        loss_function = training_params.loss_function

        traj_perms = random_batches(batch_size, 0, train_gb._num_trajectories, rng)
        t0_perms = random_batches(batch_size, 0, train_gb._num_timesteps-net_params.T, rng) 
        dropout_rng = jax.random.split(rng, batch_size)
        rng, net_rng = jax.random.split(rng)

        def loss_fn(params, batch_graphs, batch_data):
            if name == 'LC2':
                batch_control = jnp.array(batch_data[0])
                Q = jnp.array(batch_data[1]).reshape(-1,1)
                Phi = jnp.array(batch_data[2]).reshape(-1,1)
                pred_graphs = state.apply_fn(params, batch_graphs, batch_control, net_rng, rngs={'dropout': dropout_rng})
                predictions_e = pred_graphs.edges[:,:,0].squeeze()
                targets_e = jnp.concatenate((Q, Phi),axis=1).squeeze() # order is the same as LC2 graph edges
                loss = optax.squared_error(predictions_e, targets_e).mean() # MSE
            elif name == 'CoupledLC':
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
            elif name == 'Alternator':
                batch_control = jnp.array(batch_data[0])
                pred_graphs = state.apply_fn(params, batch_graphs, batch_control, net_rng, rngs={'dropout': dropout_rng})
                predictions_e = pred_graphs.edges[:,:,0].squeeze()
                targets_e = [data.reshape(-1, 1) for data in batch_data[1:-1]] # excluding control and H
                targets_e = jnp.concatenate(targets_e, axis=1).squeeze()
                loss = optax.squared_error(predictions_e, targets_e).mean()               
            else:
                batch_control = jnp.array(batch_data[0])
                t = batch_data[1]
                differential_state_targets = batch_data[2]
                algebraic_state_targets = batch_data[3]
                hamiltonian_target = batch_data[4]
                residuals_target = batch_data[5]
                pred_graphs = state.apply_fn(
                    params, batch_graphs, batch_control, t, net_rng, rngs={'dropout': dropout_rng}
                )
                predictions = train_gb.get_batch_pred_data(pred_graphs)
                differential_state_predictions = predictions[0]
                algebraic_state_predictions = predictions[1]
                hamiltonian_prediction = predictions[2]
                residuals_prediction = jnp.sum(jnp.abs(predictions[3]), axis=-1).reshape(-1, 1)
                loss = optax.squared_error(differential_state_predictions, differential_state_targets).mean() \
                        + optax.squared_error(algebraic_state_predictions, algebraic_state_targets).mean() \
                        + 0.1 * optax.squared_error(hamiltonian_prediction, hamiltonian_target).mean()
                # loss = optax.squared_error(algebraic_state_predictions, algebraic_state_targets).mean() \
                #      + 0.1 * optax.squared_error(hamiltonian_prediction, hamiltonian_target).mean() \
                #      + 0.1 * optax.squared_error(residuals_prediction, residuals_target).mean()
            return loss

        def train_batch(state, trajs, t0_idxs):
            tf_idxs = t0_idxs + net_params.T
            batch_control = train_gb.get_control(trajs, t0_idxs)
            t0s = t0_idxs * net_params.dt
            batch_exp_data = train_gb.get_exp_data(trajs, tf_idxs)
            batch_data = (batch_control, t0s, *batch_exp_data)            
            graphs = train_gb.get_graph_batch(trajs, t0_idxs)
            loss, grads = jax.value_and_grad(loss_fn)(state.params, graphs, batch_data)
            state = state.apply_gradients(grads=grads)
            return state, loss
        
        state, epoch_loss = double_scan(train_batch, state, traj_perms, t0_perms)

        train_loss = jnp.asarray(epoch_loss).mean()

        return state, TrainMetrics.single_from_model_output(loss=train_loss)

    def rollout(eval_state: TrainState, traj_idx: int, ti: int = 0):
        tf = jnp.floor_divide(training_params.rollout_timesteps + 1, net.T)
        tf_idxs = ti + jnp.arange(1, tf)
        tf_idxs = tf_idxs * net.T
        t0_idxs = tf_idxs - net_params.T
        ts = tf_idxs * net.dt
        graph = eval_gb.get_graph(traj_idx, ti)
        controls = eval_gb.get_control(traj_idx, t0_idxs)
        exp_data = eval_gb.get_exp_data(traj_idx, tf_idxs)
        get_pred_data = eval_gb.get_pred_data
        
        def forward_pass(graph, inputs):  
            control, t = inputs         
            graph = eval_state.apply_fn(eval_state.params, graph, control, t, jax.random.key(config.seed))
            pred_data = get_pred_data(graph)
            graph = graph._replace(globals=None)
            return graph, pred_data

        _, pred_data = jax.lax.scan(forward_pass, graph, (controls, t0_idxs * net.dt))
        
        losses = [
            jnp.sum(optax.l2_loss(predictions=pred_data[i], targets=exp_data[i])) for i in range(len(exp_data))
        ]
        eval_metrics = [EvalMetrics.single_from_model_output(loss=loss) for loss in losses]
        pred_data = np.concatenate(pred_data, axis=1)
        exp_data = np.concatenate(exp_data, axis=1)
        return pred_data, exp_data, eval_metrics
    
    state = TrainState.create(
        apply_fn=batched_apply,
        params=params,
        tx=tx,
    )

    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)        
    ckpt_mngr = ocp.CheckpointManager(
        dirs['ckpt'],
        options=options,
        item_handlers=ocp.StandardCheckpointHandler(),
    )

    # Restore state from checkpoint
    if restore: state = ckpt_mngr.restore(paths.ckpt_step, args=ocp.args.StandardRestore(state))

    # Create evaluation network
    eval_net = create_net(net_params)
    eval_net.training = False
    eval_state = state.replace(apply_fn=eval_net.apply)

    # Create logger to report training progress
    report_progress = periodic_actions.ReportProgress(
            num_train_steps=training_params.num_epochs,
            writer=writer
        )
    profiler = periodic_actions.Profile(logdir=dirs['log'])
    hooks = [report_progress, profiler]
    ts_size = train_gb._num_timesteps
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
    print('##################################################')
    print(f"Number of parameters: {num_parameters(params)}")
    print(f"Training system: {name}")
    print(f"Saving to {os.path.relpath(dirs['home'])}")
    print(f"\nStarting training at epoch {init_epoch}")
    print('##################################################')
    for epoch in range(init_epoch, final_epoch):
        rng, train_rng = jax.random.split(rng)
        state, metrics_update = train_epoch(state, training_params.batch_size, train_rng) 
        if train_metrics is None:
            train_metrics = metrics_update
        else:
            train_metrics = train_metrics.merge(metrics_update)

        print(f'Epoch {epoch}: ' + 'loss = {:.15}'.format(train_metrics.compute()["loss"]))

        is_last_step = (epoch == final_epoch - 1)

        if epoch % config.eval_every_steps == 0 or is_last_step:
            eval_metrics = None
            eval_state = eval_state.replace(params=state.params)

            with report_progress.timed('eval'):
                error_sums = [0] * 4
                num_eval_trajs = 2 # eval_gb._num_trajectories
                for i in range(num_eval_trajs):
                    pred_data, exp_data, eval_metrics = rollout(eval_state, traj_idx=i)
                    for j in range(len(error_sums)):
                        error_sums[j] += eval_metrics[j].compute()['loss']

                    eval_gb.plot(pred_data, exp_data, 
                                 plot_dir=os.path.join(dirs['plot'], f'traj_{i}'),
                                 prefix=f'Epoch {epoch}: eval_traj_{i}')

                rollout_error_state = np.array(error_sums) / (num_eval_trajs * eval_gb._num_timesteps)

                print('##################################################')
                print(f'Epoch {epoch} evaluation:\n \t Differential state errors {rollout_error_state[0]} \n \t Algebraic states error {rollout_error_state[1]} \n \t Hamiltonian error {rollout_error_state[2]} \n \t Sum residuals {rollout_error_state[3]}')
                print('##################################################')
                rollout_error = rollout_error_state[0] + rollout_error_state[1]
                writer.write_scalars(epoch, add_prefix_to_keys({'loss': rollout_error_state[0]}, 'eval_diff'))
                writer.write_scalars(epoch, add_prefix_to_keys({'loss': rollout_error_state[1]}, 'eval_alg'))

                if rollout_error < min_error: 
                    # Save best model
                    min_error = rollout_error
                    ckpt_step = epoch
                    print(f'Saving best model at epoch {epoch}')
                    with report_progress.timed('checkpoint'):
                        ckpt_mngr.save(epoch, args=ocp.args.StandardSave(state))
                        ckpt_mngr.wait_until_finished()

                if epoch > training_params.min_epochs: # train for at least 'min_epochs' epochs
                    early_stop = early_stop.update(rollout_error)
                    if early_stop.should_stop:
                        print(f'Met early stopping criteria, breaking at epoch {epoch}')
                        training_params.num_epochs = epoch - init_epoch
                        is_last_step = True
                if optimizing_hparams:
                    optimizing_hparams.report(rollout_error, step=epoch)

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

    print('##################################################')
    print('Training completed')
    print(f'\t Min error: {min_error.item()}')
    print('##################################################')


    return config.metrics.min_error if optimizing_hparams else config

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

    