import os
import jax
import optax
import json

import matplotlib.pyplot as plt
import jax.numpy as jnp

import ml_collections
from clu import metric_writers
from clu import periodic_actions
from clu import checkpoint

from flax.training.train_state import TrainState
from flax.training.early_stopping import EarlyStopping

import orbax.checkpoint as ocp
from flax.training import orbax_utils

from time import strftime
from timeit import default_timer
from argparse import ArgumentParser
from graph_builder import *
from scripts.models import *
from scripts.graph_nets import *
from utils.train_utils import *
from utils.data_utils import *
from utils.jax_utils import *
from utils.gnn_utils import *

from config import create_gnn_config

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.5'

def eval(config: ml_collections.ConfigDict):
    training_params = config.training_params
    net_params = config.net_params
    paths = config.paths
    
    name = set_name(config)
    log_dir = os.path.join(paths.dir, 'log')
    checkpoint_dir = os.path.join(paths.dir, 'checkpoint')
    plot_dir = os.path.join(paths.dir, 'eval_plots')

    rng = jax.random.key(training_params.seed)
    rng, init_rng, net_rng = jax.random.split(rng, 3)

    # Create writer for logs
    writer = metric_writers.create_default_writer(logdir=log_dir)

    tx = optax.adam(**config.optimizer_params)

    graph_builder = create_graph_builder(name, training_params, net_params)

    train_gb = graph_builder(paths.training_data_path)
    eval_gb = graph_builder(paths.evaluation_data_path)
    
    t0 = 0
    if name == 'MassSpring':
        t0 = net_params.vel_history
        net_params.norm_stats = train_gb._norm_stats
        init_control = eval_gb._control[0, 0]
    elif name == 'LC2':
        init_control = jnp.array(eval_gb._Volt[0])
    elif 'LC' in name:
        init_control = jnp.array(0)

    net_params.system_params = train_gb.system_params

    eval_net = create_net(name, training_params, net_params)
    eval_net.training = False
    init_graph = eval_gb.get_graph(traj_idx=0, t=t0)
    params = eval_net.init(init_rng, 0, init_graph, init_control, net_rng)
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
        tf_idxs = (ti + jnp.arange(1, (training_params.rollout_timesteps + 1) // eval_net.num_mp_steps)) * eval_net.num_mp_steps
        ti = round(t0 / eval_net.num_mp_steps) * eval_net.num_mp_steps
        tf_idxs = jnp.unique(tf_idxs.clip(min=ti + eval_net.num_mp_steps, max=eval_gb._num_timesteps))
        t0_idxs = tf_idxs - eval_net.num_mp_steps
        ts = tf_idxs * eval_net.dt
        graph = eval_gb.get_graph(traj_idx, ti)

        if name == 'MassSpring':
            controls = eval_gb._control[traj_idx, t0_idxs]
            exp_qs_buffer = eval_gb._qs[traj_idx, tf_idxs]
            exp_as_buffer = eval_gb._accs[traj_idx, tf_idxs]
        elif name == 'LC':
            controls = tf_idxs - eval_net.num_mp_steps # used as 'length' for scan loop
            exp_Q = eval_gb._Q[traj_idx, tf_idxs]
            exp_Phi = eval_gb._Phi[traj_idx, tf_idxs]
            exp_H = eval_gb._H[traj_idx, tf_idxs]
            exp_data = (exp_Q, exp_Phi, exp_H)
        elif name == 'LC1':
            controls = eval_gb._control[traj_idx, t0_idxs]
            exp_Q1 = eval_gb._Q1[traj_idx, tf_idxs]
            exp_Phi1 = eval_gb._Phi1[traj_idx, tf_idxs]
            exp_Q3 = eval_gb._Q3[traj_idx, tf_idxs]
            exp_H = eval_gb._H[traj_idx, tf_idxs]
            exp_data = (exp_Q1, exp_Phi1, exp_Q3, exp_H)
        elif name == 'LC2':
            controls = eval_gb._control[traj_idx, t0_idxs]
            exp_Phi = eval_gb._Phi[traj_idx, tf_idxs]
            exp_Q = eval_gb._Q[traj_idx, tf_idxs]
            exp_H = eval_gb._H[traj_idx, tf_idxs]
            exp_data = (exp_Phi, exp_Q, exp_H)
        elif name == 'CoupledLC':
            controls = t0_idxs # used as 'length' for scan loop
            exp_Q1 = eval_gb._Q1[traj_idx, tf_idxs]
            exp_Phi1 = eval_gb._Phi1[traj_idx, tf_idxs]
            exp_Q2 = eval_gb._Q2[traj_idx, tf_idxs]
            exp_Phi2 = eval_gb._Phi2[traj_idx, tf_idxs]
            exp_H = eval_gb._H[traj_idx, tf_idxs]
            exp_data = (exp_Q1, exp_Phi1, exp_Q2, exp_Phi2, exp_H)
        
        def forward_pass(graph, control):
            if name == 'MassSpring':
                graph = eval_state.apply_fn(eval_state.params, graph, control, jax.random.key(0))
                pred_qs = (graph.nodes[:,0]).squeeze()
                pred_accs = (graph.nodes[:,-1]).squeeze()
                graph = graph._replace(nodes=graph.nodes[:,:-1]) # remove acceleration  
                return graph, (pred_qs, pred_accs)
            
            elif name == 'LC':
                graph = eval_state.apply_fn(eval_state.params, graph, None, jax.random.key(0))
                pred_Q = (graph.edges[0]).squeeze()
                pred_Phi = (graph.edges[1]).squeeze()
                pred_H = (graph.globals).squeeze()
                graph = graph._replace(globals=None)
                return graph, (pred_Q, pred_Phi, pred_H)
            
            elif name == 'LC1':
                graph = eval_state.apply_fn(eval_state.params, traj_idx, graph, control, jax.random.key(0))
                pred_Q1 = (graph.edges[0,1]).squeeze()
                pred_Phi1 = (graph.edges[1,0]).squeeze()
                pred_Q3 = (graph.edges[2],1).squeeze()
                pred_H = (graph.globals).squeeze()
                graph = graph._replace(globals=None)
                return graph, (pred_Q1, pred_Phi1, pred_Q3, pred_H)
            
            elif name == 'LC2':
                graph = eval_state.apply_fn(eval_state.params, traj_idx, graph, control, jax.random.key(0))
                pred_Phi = (graph.edges[1,0]).squeeze()
                pred_Q = (graph.edges[2,1]).squeeze()
                pred_H = (graph.globals).squeeze()
                graph = graph._replace(globals=None)
                return graph, (pred_Phi, pred_Q, pred_H)
            
            elif name == 'CoupledLC':
                graph = eval_state.apply_fn(eval_state.params, traj_idx, graph, None, jax.random.key(0))
                pred_Q1 = (graph.edges[0,1]).squeeze()
                pred_Phi1 = (graph.edges[1,0]).squeeze()
                pred_Q3 = (graph.edges[2,1]).squeeze()
                pred_Phi2 = (graph.edges[3,0]).squeeze()
                pred_Q2 = (graph.edges[4,1]).squeeze()
                pred_H = (graph.globals).squeeze()
                graph = graph._replace(globals=None)
                return graph, (pred_Q1, pred_Phi1, pred_Q2, pred_Phi2, pred_H)
            
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
            losses = [optax.l2_loss(predictions=pred_data[i], targets=exp_data[i]) for i in range(len(exp_data))]
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
        
        plot_evaluation_curves(ts, pred_data, exp_data, aux_data,
                                plot_dir=os.path.join(plot_dir, f'traj_{i}'),
                                prefix=f'eval_traj_{i}')
        writer.write_scalars(i, add_prefix_to_keys({'loss': sum(error_sums)}, 'eval'))
        
    rollout_error = np.Inf
    if name == 'MassSpring':
        rollout_error = rollout_error_sum / eval_gb._num_trajectories
        print(f'Rollout mean position loss = {jnp.round(rollout_error, 4)}')
        
    elif 'LC' in name:
        rollout_mean_error = np.array(error_sums) / eval_gb._num_trajectories
        print(f'State errors {rollout_mean_error}')
        rollout_error = sum(rollout_mean_error)

    # Save evaluation metrics to json
    eval_metrics = {
        'mean_rollout_error': (rollout_error / eval_gb._num_trajectories).tolist(),
        'evaluation_data_path': paths.evaluation_data_path,
    }
    eval_metrics_file = os.path.join(plot_dir, 'eval_metrics.js')
    with open(eval_metrics_file, "w") as outfile:
        json.dump(eval_metrics, outfile)

def train(config: ml_collections.ConfigDict):
    training_params = config.training_params
    net_params = config.net_params
    paths = config.paths
        
    name = set_name(config)
    
    restore = False if paths.dir == None else True

    if paths.dir == None:
        config.paths.dir = os.path.join(os.curdir, f'results/{training_params.net_name}/{config.system_name}/{strftime("%m%d-%H%M")}_{training_params.trial_name}')
        paths.dir = config.paths.dir
    
    log_dir = os.path.join(paths.dir, 'log')
    plot_dir = os.path.join(paths.dir, 'plots')
    checkpoint_dir = os.path.join(paths.dir, 'checkpoint')
    checkpoint_dir = os.path.join(checkpoint_dir, 'best_model')
    checkpoint_dir = os.path.abspath(checkpoint_dir)

    rng = jax.random.key(training_params.seed)
    rng, init_rng, net_rng = jax.random.split(rng, 3)

    # Create writer for logs
    writer = metric_writers.create_default_writer(logdir=log_dir)
    # Create optimizer
    tx = optax.adam(**config.optimizer_params)
    # Create training and evaluation data loaders

    graph_builder = create_graph_builder(name, training_params, net_params)
    train_gb = graph_builder(paths.training_data_path)
    eval_gb = graph_builder(paths.evaluation_data_path)

    t0 = 0
    if name == 'MassSpring':
        net_params.norm_stats = train_gb._norm_stats
        t0 = net_params.vel_history
        init_control = train_gb._control[0, t0, :]
    elif 'LC' in name:
        init_control = train_gb._control[0,0]
        net_params.system_params = train_gb.system_params

    # Initialize training network
    init_graph = train_gb.get_graph(traj_idx=0, t=t0)
    net = create_net(name, training_params, net_params)
    params = net.init(init_rng, 0, init_graph, init_control, net_rng)
    if name == 'MassSpring':
        batched_apply = jax.vmap(net.apply, in_axes=(None, 0, 0, None))
    else:
        batched_apply = jax.vmap(net.apply, in_axes=(None, 0, 0, 0, None))

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
        loss_function = training_params.loss_function

        traj_perms = random_batch(batch_size, 0, train_gb._num_trajectories, rng)
        t0_perms = random_batch(batch_size, t0, train_gb._num_timesteps-time_offset, rng) 
        dropout_rng = jax.random.split(rng, batch_size)
        rng, net_rng = jax.random.split(rng)

        def loss_fn(params, batch_graphs, batch_data):
            if loss_function == 'acceleration':
                batch_targets, batch_control = batch_data
                pred_graphs = state.apply_fn(params, batch_graphs, batch_control, net_rng, rngs={'dropout': dropout_rng})
                predictions = pred_graphs.nodes[:,:,-1] 
                loss = int(1e6) * optax.l2_loss(predictions=predictions, targets=batch_targets).mean()
            elif loss_function == 'state':
                batch_pos, batch_vel, batch_control = batch_data
                pred_graphs = state.apply_fn(params, batch_graphs, batch_control, net_rng, rngs={'dropout': dropout_rng})
                pos_predictions = pred_graphs.nodes[:,:,0]
                vel_predictions = pred_graphs.nodes[:,:,t0]
                loss = int(1e6) * (optax.l2_loss(predictions=pos_predictions, targets=batch_pos).mean() \
                     + optax.l2_loss(predictions=vel_predictions, targets=batch_vel).mean())
            elif name == 'LC' and loss_function == 'lc_state':
                """ l2_loss([Q_hat, Flux_hat], [Q, Flux])"""
                traj_idx = jnp.array(batch_data[0])
                Q = jnp.array(batch_data[1]).reshape(1,-1)
                Phi = jnp.array(batch_data[2]).reshape(1,-1)
                pred_graphs = state.apply_fn(params, traj_idx, batch_graphs, None, net_rng, rngs={'dropout': dropout_rng})
                predictions = pred_graphs.edges.squeeze()
                targets = jnp.concatenate((Q, Phi)).squeeze() 
                loss = optax.l2_loss(predictions, targets)
                loss = jnp.sum(loss)
            elif name == 'LC1' and loss_function == 'lc_state':
                traj_idx = jnp.array(batch_data[0])
                batch_control = jnp.array(batch_data[1])
                # Q1 = jnp.array(batch_data[2]).reshape(-1,1)
                # Phi1 = jnp.array(batch_data[3]).reshape(-1,1)
                # Q3 = jnp.array(batch_data[4]).reshape(-1,1)
                # V2 = jnp.array(batch_data[5]).reshape(-1,1)
                # V3 = jnp.array(batch_data[6]).reshape(-1,1)
                dQ1 = jnp.array(batch_data[2]).reshape(-1,1)
                Q1 = jnp.array(batch_data[3]).reshape(-1,1)
                Phi1 = jnp.array(batch_data[4]).reshape(-1,1)
                dPhi1 = jnp.array(batch_data[5]).reshape(-1,1)
                dQ3 = jnp.array(batch_data[6]).reshape(-1,1)
                Q3 = jnp.array(batch_data[7]).reshape(-1,1)
                V2 = jnp.array(batch_data[8]).reshape(-1,1)
                V3 = jnp.array(batch_data[9]).reshape(-1,1)
                pred_graphs = state.apply_fn(params, traj_idx, batch_graphs, batch_control, net_rng, rngs={'dropout': dropout_rng})

                predictions_e = jnp.column_stack((pred_graphs.edges[:,0,1], pred_graphs.edges[:,1,0], pred_graphs.edges[:,2,1]))
                targets_e = jnp.concatenate((Q1, Phi1, Q3), axis=1)

                # predictions_e = pred_graphs.edges[:,:-1].squeeze() # exclude control
                # targets_e1 = jnp.concatenate((dQ1, Phi1, dQ3), axis=1)
                # targets_e2 = jnp.concatenate((Q1, dPhi1, Q3), axis=1)
                # targets_e1 = jnp.expand_dims(targets_e1, axis=-1)
                # targets_e2 = jnp.expand_dims(targets_e2, axis=-1)
                # targets_e = jnp.concatenate((targets_e1, targets_e2), axis=-1)
                predictions_n = pred_graphs.nodes[:,1:].squeeze() # excluding ground node
                targets_n = jnp.concatenate((V2, V3), axis=1).squeeze()
                loss_e = optax.l2_loss(predictions_e, targets_e)
                loss_n = optax.l2_loss(predictions_n, targets_n)
                loss = jnp.sum(loss_e) + jnp.sum(loss_n)
            elif name == 'LC2' and loss_function == 'lc_state':
                traj_idx = jnp.array(batch_data[0])
                batch_control = jnp.array(batch_data[1])
                Phi = jnp.array(batch_data[2]).reshape(-1,1)
                Q = jnp.array(batch_data[3]).reshape(-1,1)
                Vc = jnp.array(batch_data[4]).reshape(-1,1)
                pred_graphs = state.apply_fn(params, traj_idx, batch_graphs, batch_control, net_rng, rngs={'dropout': dropout_rng})
                # predictions = pred_graphs.edges[:,jnp.array([1,2])].squeeze()
                predictions_e = jnp.column_stack((pred_graphs.edges[:,1,0], pred_graphs.edges[:,2,1]))
                predictions_n = pred_graphs.nodes[:,1].squeeze()
                targets_e = jnp.concatenate((Phi, Q),axis=1).squeeze() # order is the same as LC2 graph edges
                targets_n = Vc.squeeze()
                loss_e = optax.l2_loss(predictions_e, targets_e)
                loss_n = optax.l2_loss(predictions_n, targets_n)
                loss = jnp.sum(loss_e) + jnp.sum(loss_n)
            elif name == 'CoupledLC' and loss_function == 'lc_state':
                traj_idx = jnp.array(batch_data[0])
                Q1 = jnp.array(batch_data[1]).reshape(-1,1)
                Phi1 = jnp.array(batch_data[2]).reshape(-1,1)
                Q3 = jnp.array(batch_data[3]).reshape(-1,1)
                Phi2 = jnp.array(batch_data[4]).reshape(-1,1)
                Q2 = jnp.array(batch_data[5]).reshape(-1,1)
                V2 = jnp.array(batch_data[6]).reshape(-1,1)
                V3 = jnp.array(batch_data[7]).reshape(-1,1)
                V4 = jnp.array(batch_data[8]).reshape(-1,1)
                pred_graphs = state.apply_fn(params, traj_idx, batch_graphs, None, net_rng, rngs={'dropout': dropout_rng})
                predictions_e = jnp.column_stack((pred_graphs.edges[:,0,1], pred_graphs.edges[:,1,0],
                pred_graphs.edges[:,2,1], pred_graphs.edges[:,3,0], pred_graphs.edges[:,4,1]))
                targets_e = jnp.concatenate((Q1, Phi1, Q3, Phi2, Q2),axis=1).squeeze()
                loss_e = optax.l2_loss(predictions_e, targets_e)
                predictions_n = pred_graphs.nodes[:,1:].squeeze()
                targets_n = jnp.concatenate((V2, V3, V4),axis=1).squeeze()
                loss_n = optax.l2_loss(predictions_n, targets_n)
                loss = jnp.sum(loss_e) + jnp.sum(loss_n)
            elif name == 'LC1' and loss_function == 'hamiltonian':
                traj_idx = jnp.array(batch_data[0])
                batch_control = jnp.array(batch_data[1])
                H = jnp.array(batch_data[2])
                Q1 = jnp.array(batch_data[3]).reshape(-1,1)
                Phi1 = jnp.array(batch_data[4]).reshape(-1,1)
                Q3 = jnp.array(batch_data[5]).reshape(-1,1)
                V2 = jnp.array(batch_data[6]).reshape(-1,1)
                V3 = jnp.array(batch_data[7]).reshape(-1,1)
                pred_graphs = state.apply_fn(params, traj_idx, batch_graphs, batch_control, net_rng, rngs={'dropout': dropout_rng})
                predictions_H = pred_graphs.globals
                targets_H = H
                predictions_E = pred_graphs.edges[:,:-1].squeeze() # excluding self loop
                targets_E = jnp.concatenate((Q1, Phi1, Q3), axis=1)
                predictions_V = pred_graphs.nodes[:,1:].squeeze()
                targets_V = jnp.concatenate((V2, V3),axis=1) # excluding ground node
                loss_H = optax.l2_loss(predictions_H, targets_H)
                loss_E = optax.l2_loss(predictions_E, targets_E)
                loss_V = optax.l2_loss(predictions_V, targets_V)
                loss = jnp.sum(loss_H) + jnp.sum(loss_E) + 0.1 * jnp.sum(loss_V)
            return loss

        def train_batch(state, trajs, t0s):
            tfs = t0s + time_offset
            if loss_function == 'acceleration':
                batch_control = train_gb._control[trajs, t0s]
                batch_accs = train_gb._accs[trajs, tfs]
                batch_data = (batch_accs, batch_control)
            elif loss_function == 'state':
                batch_control = train_gb._control[trajs, t0s]
                batch_pos = train_gb._qs[trajs, tfs]
                batch_vel = train_gb._vs[trajs, tfs]
                batch_data = (batch_pos, batch_vel, batch_control)
            elif name == 'LC' and loss_function == 'lc_state':
                batch_data = (trajs, train_gb._Q[trajs, tfs], train_gb._Phi[trajs, tfs])
            elif name == 'LC1' and loss_function == 'lc_state':
                # batch_data = (trajs, train_gb._control[trajs, t0s], train_gb._Q1[trajs, tfs], train_gb._Phi1[trajs, tfs], train_gb._Q3[trajs, tfs], train_gb._V2[trajs, tfs], train_gb._V3[trajs, tfs])
                batch_data = (trajs, train_gb._control[trajs, t0s], train_gb._dQ1[trajs, tfs], train_gb._Q1[trajs, tfs], train_gb._Phi1[trajs, tfs], train_gb._dPhi1[trajs, tfs], train_gb._dQ3[trajs, tfs], train_gb._Q3[trajs, tfs], train_gb._V2[trajs, tfs], train_gb._V3[trajs, tfs])
            elif name == 'LC2' and loss_function == 'lc_state':
                batch_data = (trajs, train_gb._control[trajs, t0s], train_gb._Phi[trajs, tfs], train_gb._Q[trajs, tfs], train_gb._Vc[trajs, tfs])
            elif name == 'CoupledLC' and loss_function == 'lc_state':
                batch_data = (trajs, train_gb._Q1[trajs, tfs], train_gb._Phi1[trajs, tfs], 
                              train_gb._Q3[trajs, tfs], train_gb._Phi2[trajs, tfs], train_gb._Q2[trajs, tfs], train_gb._V2[trajs, tfs], train_gb._V3[trajs, tfs], train_gb._V4[trajs, tfs])
            elif name == 'LC1' and loss_function == 'hamiltonian':
                batch_data = (trajs, train_gb._control[trajs, t0s], train_gb._H[trajs, tfs], train_gb._Q1[trajs, tfs], train_gb._Phi1[trajs, tfs], train_gb._Q3[trajs, tfs], train_gb._V2[trajs, tfs], train_gb._V3[trajs, tfs]) 
            
            graphs = train_gb.get_graph_batch(trajs, t0s)
            loss, grads = jax.value_and_grad(loss_fn)(state.params, graphs, batch_data)
            state = state.apply_gradients(grads=grads)
            return state, loss
        
        state, epoch_loss = double_scan(train_batch, state, traj_perms, t0_perms)

        train_loss = jnp.asarray(epoch_loss).mean()

        return state, TrainMetrics.single_from_model_output(loss=train_loss)

    def rollout(eval_state: TrainState, traj_idx: int, ti: int = 0):
        tf_idxs = (ti + jnp.arange(1, (training_params.rollout_timesteps + 1) // net.num_mp_steps)) * net.num_mp_steps
        ti = round(t0 /  net.num_mp_steps) * net.num_mp_steps
        tf_idxs = jnp.unique(tf_idxs.clip(min=ti + net.num_mp_steps, max=eval_gb._num_timesteps))
        t0_idxs = tf_idxs - ti
        ts = tf_idxs * net.dt
        graph = eval_gb.get_graph(traj_idx, ti)

        if name == 'MassSpring':
            controls = eval_gb._control[traj_idx, t0_idxs]
            exp_qs_buffer = eval_gb._qs[traj_idx, tf_idxs]
            exp_as_buffer = eval_gb._accs[traj_idx, tf_idxs]
        elif name == 'LC':
            controls = t0_idxs # used as 'length' for scan loop
            exp_Q = eval_gb._Q[traj_idx, tf_idxs]
            exp_Phi = eval_gb._Phi[traj_idx, tf_idxs]
            exp_H = eval_gb._H[traj_idx, tf_idxs]
            exp_data = (exp_Q, exp_Phi, exp_H)
        elif name == 'LC1':
            controls = eval_gb._control[traj_idx, t0_idxs]
            exp_dQ1 = eval_gb._dQ1[traj_idx, tf_idxs]
            exp_Q1 = eval_gb._Q1[traj_idx, tf_idxs]
            exp_Phi1 = eval_gb._Phi1[traj_idx, tf_idxs]
            exp_dPhi1 = eval_gb._dPhi1[traj_idx, tf_idxs]
            exp_dQ3 = eval_gb._dQ3[traj_idx, tf_idxs]
            exp_Q3 = eval_gb._Q3[traj_idx, tf_idxs]
            exp_H = eval_gb._H[traj_idx, tf_idxs]
            # exp_data = (exp_dQ1, exp_Q1, exp_Phi1, exp_dPhi1, exp_dQ3, exp_Q3, exp_H)
            exp_data = (exp_Q1, exp_Phi1, exp_Q3, exp_H)    
        elif name == 'LC2':
            controls = eval_gb._control[traj_idx, t0_idxs]
            exp_Phi = eval_gb._Phi[traj_idx, tf_idxs]
            exp_Q = eval_gb._Q[traj_idx, tf_idxs]
            exp_H = eval_gb._H[traj_idx, tf_idxs]
            exp_data = (exp_Phi, exp_Q, exp_H)
        elif name == 'CoupledLC':
            controls = t0_idxs # used as 'length' for scan loop # TODO
            exp_Q1 = eval_gb._Q1[traj_idx, tf_idxs]
            exp_Phi1 = eval_gb._Phi1[traj_idx, tf_idxs]
            exp_Q2 = eval_gb._Q2[traj_idx, tf_idxs]
            exp_Phi2 = eval_gb._Phi2[traj_idx, tf_idxs]
            exp_H = eval_gb._H[traj_idx, tf_idxs]
            exp_data = (exp_Q1, exp_Phi1, exp_Q2, exp_Phi2, exp_H)
        
        def forward_pass(graph, control):
            if name == 'MassSpring':
                graph = eval_state.apply_fn(eval_state.params, graph, control, jax.random.key(0))
                pred_qs = (graph.nodes[:,0]).squeeze()
                pred_accs = (graph.nodes[:,-1]).squeeze()
                graph = graph._replace(nodes=graph.nodes[:,:-1]) # remove acceleration  
                return graph, (pred_qs, pred_accs)
            
            elif name == 'LC':
                graph = eval_state.apply_fn(eval_state.params, jnp.array(traj_idx), graph, None, jax.random.key(0))
                pred_Q = (graph.edges[0]).squeeze()
                pred_Phi = (graph.edges[1]).squeeze()
                pred_H = (graph.globals).squeeze()
                graph = graph._replace(globals=None)
                return graph, (pred_Q, pred_Phi, pred_H)
            
            elif name == 'LC1':
                graph = eval_state.apply_fn(eval_state.params, jnp.array(traj_idx), graph, jnp.array(0.0), jax.random.key(0)) # TODO: change from jnp.array(0.0) to control
                pred_dQ1 = (graph.edges[0,0]).squeeze()
                pred_Q1 = (graph.edges[0,1]).squeeze()
                pred_Phi1 = (graph.edges[1,0]).squeeze()
                pred_dPhi1 = (graph.edges[1,1]).squeeze()
                pred_dQ3 = (graph.edges[2,0]).squeeze()
                pred_Q3 = (graph.edges[2,1]).squeeze()
                pred_H = (graph.globals).squeeze()
                graph = graph._replace(globals=None)
                # return graph, (pred_dQ1, pred_Q1, pred_Phi1, pred_dPhi1, pred_dQ3, pred_Q3, pred_H)
                return graph, (pred_Q1, pred_Phi1, pred_Q3, pred_H) # TODO: try plotting everything         
            
            elif name == 'LC2':
                graph = eval_state.apply_fn(eval_state.params, jnp.array(traj_idx), graph, control, jax.random.key(0))
                pred_Phi = (graph.edges[1,0]).squeeze()
                pred_Q = (graph.edges[2,1]).squeeze()
                pred_H = (graph.globals).squeeze()
                graph = graph._replace(globals=None)
                return graph, (pred_Phi, pred_Q, pred_H)
            
            elif name == 'CoupledLC':
                graph = eval_state.apply_fn(eval_state.params, jnp.array(traj_idx), graph, None, jax.random.key(0))
                pred_Q1 = (graph.edges[0,1]).squeeze()
                pred_Phi1 = (graph.edges[1,0]).squeeze()
                pred_Q3 = (graph.edges[2,1]).squeeze()
                pred_Phi2 = (graph.edges[3,0]).squeeze()
                pred_Q2 = (graph.edges[4,1]).squeeze()
                pred_H = (graph.globals).squeeze()
                graph = graph._replace(globals=None)
                return graph, (pred_Q1, pred_Phi1, pred_Q2, pred_Phi2, pred_H)

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
        elif 'LC' in name:
            losses = [optax.l2_loss(predictions=pred_data[i], targets=exp_data[i]) for i in range(len(exp_data))]
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
    if restore: state = ckpt_mngr.restore(config.ckpt_step, args=ocp.args.StandardRestore(state))

    # Create evaluation network
    eval_net = create_net(name, training_params, net_params)
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
    ts_size = (train_gb._num_timesteps - t0)
    trajs_size = train_gb._num_trajectories
    steps_per_epoch = int((ts_size * trajs_size) // (training_params.batch_size ** 2))
    train_fn = train_epoch

    # Setup training epochs
    init_epoch = state.step // steps_per_epoch + 1
    final_epoch = init_epoch + training_params.num_epochs
    training_params.num_epochs = final_epoch

    early_stop = EarlyStopping(min_delta=1e-3, patience=2)

    train_metrics = None
    min_error = jnp.inf
    print(f"Start training at epoch {init_epoch}")
    for epoch in range(init_epoch, final_epoch):
        # print(f'State step on epoch {epoch}: {state.step}, {state.step // steps_per_epoch + 1}')
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
            eval_state = eval_state.replace(params=state.params)

            with report_progress.timed('eval'):
                rollout_error_sum = 0
                error_sums = [0] * train_gb._num_states
                for i in range(2): # TODO: was eval_gb._num_trajectories
                    if name == 'MassSpring':
                        ts, pred_data, exp_data, aux_data, eval_metrics = rollout(eval_state, traj_idx=i)
                        rollout_error_sum += eval_metrics.compute()['loss']   
                    elif 'LC' in name:
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
                    print(f'Epoch {epoch}: rollout mean position loss = {jnp.round(rollout_error, 4)}')
                    
                elif 'LC' in name:
                    rollout_mean_error = np.array(error_sums) / eval_gb._num_trajectories
                    print(f'Epoch {epoch} state errors {rollout_mean_error}')
                    rollout_error = sum(rollout_mean_error)

                writer.write_scalars(epoch, add_prefix_to_keys({'loss': rollout_error}, 'eval'))

                if rollout_error < min_error: 
                    # Save best model
                    min_error = rollout_error
                    print(f'Saving best model at epoch {epoch}')
                    with report_progress.timed('checkpoint'):
                        ckpt_mngr.save(epoch, args=ocp.args.StandardSave(state))
                        ckpt_mngr.wait_until_finished()

                if epoch > training_params.min_epochs: # train at least for 'min_epochs' epochs
                    early_stop = early_stop.update(rollout_error)
                    if early_stop.should_stop:
                        print(f'Met early stopping criteria, breaking at epoch {epoch}')
                        training_params.num_epochs = epoch - init_epoch
                        is_last_step = True

        if epoch % training_params.log_every_steps == 0 or is_last_step:
            writer.write_scalars(epoch, add_prefix_to_keys(train_metrics.compute(), 'train'))
            train_metrics = None

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
    parser.add_argument('--system', type=str, default='lc')
    args = parser.parse_args()

    config = create_gnn_config(args)

    if args.eval:
        eval(config)
    else:
        train(config)