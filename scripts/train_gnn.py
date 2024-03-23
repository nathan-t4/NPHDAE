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
from argparse import ArgumentParser
from graph_builder import DMSDGraphBuilder
from scripts.models import *
from utils.data_utils import *
from utils.jax_utils import *
from utils.train_utils import *

def create_gnn_config(args) -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.paths = ml_collections.ConfigDict({
        'dir': args.dir,
        'training_data_path': 'results/double_mass_spring_data/train_500_0.1.pkl',
        'evaluation_data_path': 'results/double_mass_spring_data/val_20_mass.pkl',
    })
    config.training_params = ml_collections.ConfigDict({
        'net_name': 'GraphNet',
        'num_epochs': int(5e2),
        'batch_size': 2,
        'rollout_timesteps': 1500,
        'log_every_steps': 1,
        'eval_every_steps': 10,
        'checkpoint_every_steps': 100,
        'clear_cache_every_steps': 10,
        'add_undirected_edges': True,
        'add_self_loops': True,
        'train_multi_trajectories': True,
    })
    config.optimizer_params = ml_collections.ConfigDict({
        # 'learning_rate': optax.exponential_decay(init_value=1e-3, transition_steps=5e2, decay_rate=0.1, end_value=1e-5),
        # 'learning_rate': optax.cosine_decay_schedule(init_value=1e-3, decay_steps=500, alpha=1e-5)
        'learning_rate': 1e-3,
    })
    config.net_params = ml_collections.ConfigDict({
        'prediction': 'acceleration',
        'integration_method': 'SemiImplicitEuler', 
        # 'horizon': 5, # for gnode only
        'vel_history': 5,
        'num_mp_steps': 1, # too big causes oversmoothing
        'noise_std': 0.0003,
        'latent_size': 64,
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
        work_dir = os.path.join(os.curdir, f'results/test_models/{strftime("%m%d")}_test_gnn/generalization_32_{strftime("%H%M%S")}')
    else:
        work_dir = paths.dir
            
    log_dir = os.path.join(work_dir, 'log')
    checkpoint_dir = os.path.join(work_dir, 'checkpoint')
    plot_dir = os.path.join(work_dir, 'plots')

    rng = jax.random.key(0)
    rng, init_rng, net_rng = jax.random.split(rng, 3)

    # Create writer for logs
    writer = metric_writers.create_default_writer(logdir=log_dir)

    tx = optax.adam(**config.optimizer_params)
    train_gb = DMSDGraphBuilder(paths.training_data_path, 
                                training_params.add_undirected_edges, 
                                training_params.add_self_loops, 
                                net_params.prediction, 
                                net_params.vel_history)
    eval_gb = DMSDGraphBuilder(paths.evaluation_data_path, 
                               training_params.add_undirected_edges, 
                               training_params.add_self_loops, 
                               net_params.prediction, 
                               net_params.vel_history)
    
    net_params.norm_stats = train_gb._norm_stats
    net = create_net()
    init_graph = train_gb.get_graph(traj_idx=0, t=net_params.vel_history+1)
    params = net.init(init_rng, init_graph, net_rng)
    batched_apply = jax.vmap(net.apply, in_axes=(None,0,None))

    print(f"Number of parameters {num_parameters(params)}")
    time_offset = net.horizon if training_params.net_name == 'gnode' else net.num_mp_steps

    def random_batch(batch_size: int, min: int, max: int, rng: jax.Array):
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

        def loss_fn(params, batch_graphs, batch_targets):
            pred_graphs = state.apply_fn(params, batch_graphs, net_rng, rngs={'dropout': dropout_rng})
            if net_params.prediction == 'acceleration':
                predictions = pred_graphs.nodes[:,:,-1]
            elif net_params.prediction == 'position':
                predictions = pred_graphs.nodes[:,:,0]
            loss = int(1e6) * optax.l2_loss(predictions=predictions, targets=batch_targets).mean()
            return loss

        def train_batch(state, trajs, t0s):
            tfs = t0s + time_offset
            batch_accs = train_gb._accs[trajs, tfs]
            batch_data = batch_accs
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
        def loss_fn(params, batch_graphs, batch_targets):
            pred_graphs = state.apply_fn(params, batch_graphs, net_rng, rngs={'dropout': dropout_rng})
            predictions = pred_graphs.nodes[:,:,-1]
            loss = int(1e6) * optax.l2_loss(predictions=predictions, targets=batch_targets).mean()
            return loss
        
        def train_batch(state, t0s):
            tfs = t0s + time_offset
            batch_accs = train_gb._accs[traj_idx, tfs]
            batch_data = batch_accs
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
        tf_idxs = jnp.unique(tf_idxs.clip(min=t0 + net.num_mp_steps, max=1501))
        ts = tf_idxs * net.dt

        exp_qs_buffer = eval_gb._qs[traj_idx, tf_idxs]
        exp_as_buffer = eval_gb._accs[traj_idx, tf_idxs]
        graphs = eval_gb.get_graph(traj_idx, t0)
        batched_graph = pytrees_stack([graphs])
        def forward_pass(graph, x):
            graph = eval_state.apply_fn(state.params, graph, jax.random.key(0))
            pred_qs = graph.nodes[:,:,0]
            if net_params.prediction == 'acceleration':
                pred_accs = graph.nodes[:,:,-1]
                graph = graph._replace(nodes=graph.nodes[:,:,:-1]) # remove acceleration  
                return graph, (pred_qs.squeeze(), pred_accs.squeeze())
            elif net_params.prediction == 'position':
                return graph, pred_qs.squeeze()
        
        final_batched_graph, pred_data = jax.lax.scan(forward_pass, batched_graph, None, length=len(ts))

        eval_pos_loss = optax.l2_loss(predictions=pred_data[0], targets=exp_qs_buffer).mean()

        if net_params.prediction == 'acceleration':
            aux_data = eval_gb._m[traj_idx]
            return ts, np.array(pred_data), np.array((exp_qs_buffer, exp_as_buffer)), EvalMetrics.single_from_model_output(loss=eval_pos_loss), aux_data
        elif net_params.prediction == 'position':
            return ts, np.array(pred_data), np.array(exp_qs_buffer)    

    def plot(ts, pred_data, exp_data, aux_data, prefix, show=False):
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)
        if net_params.prediction == 'acceleration':
            m = np.round(aux_data, 3)
            q0 = np.round(exp_data[0,0], 3)
            a0 = np.round(exp_data[1,0], 3)
            fig, (ax1, ax2) = plt.subplots(2,1)
            fig.suptitle(f'{prefix}: Eval Error \n q0 = {q0}, a0 = {a0}')

            ax1.set_title('Position')
            ax1.plot(ts, exp_data[0] - pred_data[0], label=[f'Mass {i}' for i in range(2)])
            ax1.set_xlabel('Time [$s$]')
            ax1.set_ylabel('Position error [$m$]')
            ax1.legend()

            ax2.set_title('Acceleration')
            ax2.plot(ts, exp_data[1] - pred_data[1], label=[f'Mass {i}' for i in range(2)])
            ax2.set_xlabel('Time [$s$]')
            ax2.set_ylabel(r'Acceleration error [$\mu m/s^2$]')
            ax2.legend()

            plt.tight_layout()
            fig.savefig(os.path.join(plot_dir, f'{prefix}_error.png'))
            plt.show() if show else plt.close()

            for i in range(2):
                fig, (ax1, ax2) = plt.subplots(2,1)
                fig.suptitle(f'{prefix}: Mass {i} \n' + rf'$m_{i} = {m[i]}$')
                ax1.set_title(f'Position')
                ax1.plot(ts, pred_data[0,:,i], label='predicted')
                ax1.plot(ts, exp_data[0,:,i], label='expected')
                ax1.set_xlabel('Time [$s$]')
                ax1.set_ylabel('Position [$m$]')
                ax1.legend()

                ax2.set_title(f'Acceleration')
                ax2.plot(ts, pred_data[1,:,i], label='predicted')
                ax2.plot(ts, exp_data[1,:,i], label='expected')
                ax2.set_xlabel('Time [$s$]')
                ax2.set_ylabel(r'Acceleration [$\mu m/s^2$]')
                ax2.legend()

                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'{prefix}_mass{i}.png'))
                if show: plt.show()
                plt.close()
        
        # elif net_params.prediction == 'position':
        #     fig, ax1 = plt.subplots(1)

        #     ax1.set_title('Position')
        #     ax1.plot(ts, exp_data - pred_data, label=[f'Mass {i}' for i in range(2)])
        #     ax1.set_xlabel('Time [$s$]')
        #     ax1.set_ylabel(r'Position error [$\mu m$]')
        #     ax1.legend()
        #     plt.tight_layout()
        #     fig.savefig(os.path.join(plot_dir, f'{prefix}_error.png'))
        #     plt.show() if show else plt.close()

        #     for i in range(2):
        #         fig, ax1 = plt.subplots(1)
        #         fig.suptitle(f'{prefix}: Mass {i}')
        #         ax1.set_title(f'Position')
        #         ax1.plot(ts, pred_data[:,i], label='predicted')
        #         ax1.plot(ts, exp_data[:,i], label='expected')
        #         ax1.set_xlabel('Time [$s$]')
        #         ax1.set_ylabel(r'Position [$\mu m$]')
        #         ax1.legend()

        #         plt.tight_layout()
        #         plt.savefig(os.path.join(plot_dir, f'{prefix}_mass{i}.png'))
        #         if show: plt.show()
        #         plt.close()
        plt.close()

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
    ckpt = checkpoint.Checkpoint(checkpoint_dir)
    state = ckpt.restore_or_initialize(state)

    trajs_size = len(train_gb._data)
    ts_size = len(train_gb._data[0]) - train_gb._vel_history
    
    steps_per_epoch = ts_size // training_params.batch_size
    if training_params.train_multi_trajectories:
        steps_per_epoch *= trajs_size // training_params.batch_size
        train_fn = train_epoch
    else:
        eval_gb = train_gb
        train_fn = train_epoch_1_traj

    init_epoch = int(state.step) // steps_per_epoch + 1
    final_epoch = init_epoch + training_params.num_epochs
    training_params.num_epochs = final_epoch

    early_stop = EarlyStopping(min_delta=1, patience=5)

    train_metrics = None
    print("Start training")
    for epoch in range(init_epoch, final_epoch):
        rng, train_rng = jax.random.split(rng)
        state, metrics_update = train_fn(state, training_params.batch_size, train_rng) 
        if train_metrics is None:
            train_metrics = metrics_update
        else:
            train_metrics = train_metrics.merge(metrics_update)

        if epoch > 100: # Train for a minimum of 100 steps
            early_stop = early_stop.update(train_metrics.compute()['loss'])
            if early_stop.should_stop:
                print(f'Met early stopping criteria, breaking at epoch {epoch}')
                ckpt.save(state)
                training_params.num_epochs = final_epoch
                break

        print(f'Epoch {epoch}: loss = {round(train_metrics.compute()["loss"], 4)}')

        is_last_step = (epoch == final_epoch - 1)

        if epoch % training_params.log_every_steps == 0 or is_last_step:
            writer.write_scalars(epoch, add_prefix_to_keys(train_metrics.compute(), 'train'))
            train_metrics = None

        if epoch % training_params.eval_every_steps == 0 or is_last_step:
            eval_metrics = None
            with report_progress.timed('eval'):
                eval_state = eval_state.replace(params=state.params)
                rng, eval_rng = jax.random.split(rng)
                rng_idx = jax.random.randint(eval_rng, [1], minval=0, maxval=len(eval_gb._data)-1).item()
                if not training_params.train_multi_trajectories: rng_idx = 0
                ts, pred_data, exp_data, eval_metrics, aux_data = rollout(state, traj_idx=rng_idx)
                writer.write_scalars(epoch, add_prefix_to_keys(eval_metrics.compute(), 'eval'))
                plot(ts, pred_data, exp_data, aux_data, prefix=f'Epoch {epoch}')
            
        if epoch % training_params.checkpoint_every_steps == 0 or is_last_step:
            with report_progress.timed('checkpoint'):
                ckpt.save(state)

        if epoch % training_params.clear_cache_every_steps == 0 or is_last_step: 
            jax.clear_caches()

    # Save config to json
    config_js = config.to_json_best_effort()
    run_params_file = os.path.join(work_dir, 'run_params.js')
    with open(run_params_file, "w") as outfile:
        json.dump(config_js, outfile)

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
    
    if paths.dir == None:
        work_dir = os.path.join(os.curdir, f'results/test_models/{strftime("%m%d")}_test_gnn/test_generalization_masses_{strftime("%H%M%S")}')
    else:
        work_dir = paths.dir
            
    log_dir = os.path.join(work_dir, 'log')
    checkpoint_dir = os.path.join(work_dir, 'checkpoint')
    plot_dir = os.path.join(work_dir, 'eval_plots')

    rng = jax.random.key(0)
    rng, init_rng, net_rng = jax.random.split(rng, 3)

    # Create writer for logs
    writer = metric_writers.create_default_writer(logdir=log_dir)

    tx = optax.adam(**config.optimizer_params)
    eval_gb = DMSDGraphBuilder(paths.evaluation_data_path, 
                               training_params.add_undirected_edges, 
                               training_params.add_self_loops, 
                               net_params.prediction, 
                               net_params.vel_history)
    
    net_params.norm_stats = eval_gb._norm_stats
    eval_net = create_net()
    eval_net.training = False
    init_graph = eval_gb.get_graph(traj_idx=0, t=net_params.vel_history+1)
    params = eval_net.init(init_rng, init_graph, net_rng)
    batched_apply = jax.vmap(eval_net.apply, in_axes=(None,0,None))
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
        tf_idxs = jnp.unique(tf_idxs.clip(min=t0 + eval_net.num_mp_steps, max=1501))
        ts = tf_idxs * eval_net.dt

        exp_qs_buffer = eval_gb._qs[traj_idx, tf_idxs]
        exp_as_buffer = eval_gb._accs[traj_idx, tf_idxs]
        graphs = eval_gb.get_graph(traj_idx, t0)
        batched_graph = pytrees_stack([graphs])
        def forward_pass(graph, x):
            graph = eval_state.apply_fn(state.params, graph, jax.random.key(0))
            pred_qs = graph.nodes[:,:,0]
            if net_params.prediction == 'acceleration':
                pred_accs = graph.nodes[:,:,-1]
                graph = graph._replace(nodes=graph.nodes[:,:,:-1]) # remove acceleration  
                return graph, (pred_qs.squeeze(), pred_accs.squeeze())
            elif net_params.prediction == 'position':
                return graph, pred_qs.squeeze()
        
        final_batched_graph, pred_data = jax.lax.scan(forward_pass, batched_graph, None, length=len(ts))

        eval_pos_loss = optax.l2_loss(predictions=pred_data[0], targets=exp_qs_buffer).mean()

        if net_params.prediction == 'acceleration':
            aux_data = eval_gb._m[traj_idx]
            return ts, np.array(pred_data), np.array((exp_qs_buffer, exp_as_buffer)), aux_data, EvalMetrics.single_from_model_output(loss=eval_pos_loss)
        elif net_params.prediction == 'position':
            return ts, np.array(pred_data), np.array(exp_qs_buffer)  

    def plot(ts, pred_data, exp_data, aux_data, prefix, show=False):
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)
        if net_params.prediction == 'acceleration':
            m = np.round(aux_data, 3)
            q0 = np.round(exp_data[0,0], 3)
            a0 = np.round(exp_data[1,0], 3)
            fig, (ax1, ax2) = plt.subplots(2,1)
            fig.suptitle(f'{prefix}: Eval Error \n q0 = {q0}, a0 = {a0}')

            ax1.set_title('Position')
            ax1.plot(ts, exp_data[0] - pred_data[0], label=[f'Mass {i}' for i in range(2)])
            ax1.set_xlabel('Time [$s$]')
            ax1.set_ylabel('Position error [$m$]')
            ax1.legend()

            ax2.set_title('Acceleration')
            ax2.plot(ts, exp_data[1] - pred_data[1], label=[f'Mass {i}' for i in range(2)])
            ax2.set_xlabel('Time [$s$]')
            ax2.set_ylabel(r'Acceleration error [$\mu m/s^2$]')
            ax2.legend()

            plt.tight_layout()
            fig.savefig(os.path.join(plot_dir, f'{prefix}_error.png'))
            plt.show() if show else plt.close()

            for i in range(2):
                fig, (ax1, ax2) = plt.subplots(2,1)
                fig.suptitle(f'{prefix}: Mass {i} \n' + rf'$m_{i}$ = {m[i]}')
                ax1.set_title(f'Position')
                ax1.plot(ts, pred_data[0,:,i], label='predicted')
                ax1.plot(ts, exp_data[0,:,i], label='expected')
                ax1.set_xlabel('Time [$s$]')
                ax1.set_ylabel('Position [$m$]')
                ax1.legend()

                ax2.set_title(f'Acceleration')
                ax2.plot(ts, pred_data[1,:,i], label='predicted')
                ax2.plot(ts, exp_data[1,:,i], label='expected')
                ax2.set_xlabel('Time [$s$]')
                ax2.set_ylabel(r'Acceleration [$\mu m/s^2$]')
                ax2.legend()

                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'{prefix}_mass{i}.png'))
                if show: plt.show()
                plt.close()

    for i in range(len(eval_gb.data)):
        ts, pred_data, exp_data, aux_data, eval_metrics = rollout(state, traj_idx=i)
        writer.write_scalars(i, add_prefix_to_keys(eval_metrics.compute(), 'eval'))
        plot(ts, pred_data, exp_data,aux_data, prefix=f'eval_traj_{i}')

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