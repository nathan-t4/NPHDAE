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

from time import strftime
from argparse import ArgumentParser
from scripts.graphs import *
from scripts.models import *
from utils.data_utils import *
from utils.jax_utils import *
from utils.train_utils import *

def create_gnn_config(args):
    config = ml_collections.ConfigDict()
    config.config = ml_collections.ConfigDict({
        'dir': args.dir,
        'training_data_path': 'results/double_mass_spring_data/no_control_train.pkl',
        'evaluation_data_path': 'results/double_mass_spring_data/no_control_val.pkl',
    })
    config.training_params = ml_collections.ConfigDict({
        'num_epochs': int(1e3),
        'horizon': 5,
        'batch_size': 2,
        'rollout_timesteps': 1500,
        'eval_every_steps': 50,
        'checkpoint_every_steps': 100,
        'clear_cache_every_steps': 10,
        'add_undirected_edges': True,
        'add_self_loops': True,
    })
    config.net_params = ml_collections.ConfigDict({
        'prediction': 'acceleration',
        'integration_method': 'semi_implicit_euler', 
        'num_mp_steps': 2, # too large causes oversmoothing
        'latent_size': 128, # too large causes oversmoothing
        'hidden_layers': 2,
        'activation': 'relu',
        'use_edge_model': True,
        'layer_norm': True,
        'shared_params': False,
        'dropout_rate': 0.5, # TODO: use 0.5
        'add_undirected_edges': config.training_params.add_undirected_edges,
        'add_self_loops': config.training_params.add_self_loops,
    })
    return config

def test_graph_network(config: ml_collections.ConfigDict):
    if config.config.dir == None:
        work_dir = os.path.join(os.curdir, f'results/test_models/{strftime("%m%d")}_test_gnn/test_norm_128_128_mp_2_batch_2_{strftime("%H%M%S")}')
    else:
        work_dir = config.config.dir
            
    train_data, train_normalization_stats = load_data_jnp(config.config.training_data_path)
    eval_data, eval_normalization_stats = load_data_jnp(config.config.training_data_path)
    config.net_params.normalization_stats = train_normalization_stats

    log_dir = os.path.join(work_dir, 'log')
    checkpoint_dir = os.path.join(work_dir, 'checkpoint')
    plot_dir = os.path.join(work_dir, 'plots')

    rng = jax.random.key(0)
    rng, init_rng = jax.random.split(rng)

    # Create writer for logs
    writer = metric_writers.create_default_writer(logdir=log_dir)

    tx = optax.adam(1e-3)
    net = GraphNet(**config.net_params)
    init_graph = generate_graph_batch(data=train_data,
                                      traj_idx=0,
                                      t0s=[0],
                                      horizon=config.training_params.horizon,
                                      add_undirected_edges=config.training_params.add_undirected_edges,
                                      add_self_loops=config.training_params.add_self_loops)[0]
    params = net.init(init_rng, init_graph)
    batched_apply = jax.vmap(net.apply, in_axes=(None,0))

    state = TrainState.create(
        apply_fn=batched_apply,
        params=params,
        tx=tx,
    )

    # Create logger to report training progress
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=config.training_params.num_epochs,
        writer=writer
    )

    def random_batch(ds, batch_size, rng, axis=0):
        ds_size = ds.shape[axis]
        steps_per_epoch = ds_size // batch_size
        perms = jnp.arange(ds_size)
        perms = jax.random.permutation(rng, perms)
        perms = perms[: steps_per_epoch * batch_size].sort().reshape(-1,batch_size)
        perms = jax.random.permutation(rng, perms, axis=0)
        return perms

    def train_epoch(state, ds, batch_size, rng):
        ''' Train one epoch using all trajectories '''     
        traj_perms = random_batch(ds, batch_size, rng, axis=0)
        t0_perms = random_batch(ds, batch_size, rng, axis=1)
        t0_perms = t0_perms.clip(min=config.training_params.horizon)

        def loss_fn(params, batch_graphs, batch_targets):
            pred_graphs = state.apply_fn(params, batch_graphs)
            if config.net_params.prediction == 'acceleration':
                predictions = pred_graphs.nodes[:,:,-1]
            elif config.net_params.prediction == 'position':
                predictions = pred_graphs.nodes[:,:,0]
            loss = int(1e6) * optax.l2_loss(predictions=predictions, targets=batch_targets).mean()
            return loss

        def train_batch(state, trajs, t0s):
            if config.net_params.prediction == 'acceleration':
                batch_accs = ds[trajs, t0s, 8:11]
                batch_data = batch_accs.reshape(-1,3) 
            elif config.net_params.prediction == 'position':
                batch_qs = ds[trajs, t0s, 0:3]
                batch_data = batch_qs.reshape(-1,3)
            else:
                raise RuntimeError('Invalid prediction - train_batch')
            graphs = generate_graph_batch(data=ds, 
                                          traj_idx=trajs,
                                          t0s=t0s,
                                          horizon=config.training_params.horizon,
                                          add_undirected_edges=config.training_params.add_undirected_edges,
                                          add_self_loops=config.training_params.add_self_loops)
            batch_graph = pytrees_stack(graphs) # explicitly batch graphs
            loss, grads = jax.value_and_grad(loss_fn)(state.params, batch_graph, batch_data)
            state = state.apply_gradients(grads=grads)
            return state, loss
        
        state, epoch_loss = double_scan(train_batch, state, traj_perms, t0_perms)
        # state, epoch_loss = jax.lax.scan(train_batch, state, t0_perms)

        train_loss = jnp.asarray(epoch_loss).mean()

        return state, TrainMetrics.single_from_model_output(loss=train_loss)

    def train_epoch_1_traj(state, ds, batch_size, rng):
        ''' Train one epoch using just one trajectory (traj_idx = 0) '''     
        t0_perms = random_batch(ds, batch_size, rng, axis=1)
        t0_perms = t0_perms.clip(min=config.training_params.horizon) # for velocity history (horizon)
        epoch_loss = []

        dropout_rng = jax.random.split(rng, batch_size)
        def loss_fn(params, batch_graphs, batch_targets):
            pred_graphs = state.apply_fn(params, batch_graphs, rngs={'dropout': dropout_rng})
            predictions = pred_graphs.nodes[:,:,-1]
            loss = int(1e6) * optax.l2_loss(predictions=predictions, targets=batch_targets).mean()
            return loss
        
        def train_batch(state, t0s):
            batch_accs = ds[0, t0s, 8:11]
            batch_data = batch_accs
            graphs = generate_graph_batch(data=ds, 
                                          traj_idx=0,
                                          t0s=t0s,
                                          horizon=config.training_params.horizon,
                                          add_undirected_edges=config.training_params.add_undirected_edges,
                                          add_self_loops=config.training_params.add_self_loops)
            batch_graph = pytrees_stack(graphs) # explicitly batch graphs
            loss, grads = jax.value_and_grad(loss_fn)(state.params, batch_graph, batch_data)

            state = state.apply_gradients(grads=grads)

            return state, loss
        
        state, epoch_loss = jax.lax.scan(train_batch, state, t0_perms)
        train_loss = jnp.asarray(epoch_loss).mean()

        return state, TrainMetrics.single_from_model_output(loss=train_loss)

    def rollout(state, ds, traj_idx=0, t0=0):
        net.training = False

        t0 = jnp.array(t0).clip(min=config.training_params.horizon)
        ts = jnp.arange(t0, config.training_params.rollout_timesteps) * net.dt
        exp_qs_buffer = ds[traj_idx, t0:config.training_params.rollout_timesteps, 0:3]
        exp_as_buffer = ds[traj_idx, t0:config.training_params.rollout_timesteps, 8:11]
        graphs = generate_graph_batch(data=ds, 
                                      traj_idx=traj_idx,
                                      t0s=[t0], 
                                      horizon=config.training_params.horizon,
                                      add_undirected_edges=config.training_params.add_undirected_edges,
                                      add_self_loops=config.training_params.add_self_loops)
        batched_graph = pytrees_stack(graphs)
        
        def forward_pass(graph, x):
            graph = state.apply_fn(state.params, graph)
            pred_qs = graph.nodes[:,:,0]
            if config.net_params.prediction == 'acceleration':
                pred_accs = graph.nodes[:,:,-1]
                graph = graph._replace(nodes=graph.nodes[:,:,:-1]) # remove acceleration    
                return graph, (pred_qs.squeeze(), pred_accs.squeeze())
            elif config.net_params.prediction == 'position':
                return graph, pred_qs.squeeze()
        
        final_batched_graph, pred_data = jax.lax.scan(forward_pass, batched_graph, None, length=config.training_params.rollout_timesteps - config.training_params.horizon)
        if config.net_params.prediction == 'acceleration':
            return ts, np.array(pred_data), np.array((exp_qs_buffer, exp_as_buffer))
        elif config.net_params.prediction == 'position':
            return ts, np.array(pred_data), np.array(exp_qs_buffer)        

    def plot(ts, pred_data, exp_data, prefix, show=False):
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)
        if config.net_params.prediction == 'acceleration':
            fig, (ax1, ax2) = plt.subplots(2,1)

            ax1.set_title('Position')
            ax1.plot(ts, exp_data[0] - pred_data[0], label=[f'Mass {i}' for i in range(3)])
            ax1.set_xlabel('Time [$s$]')
            ax1.set_ylabel('Position error [$m$]')
            # ax1.set_ylim([-0.5,0.5])
            ax1.legend()

            fig.suptitle(f'{prefix}: Error from rollout')
            ax2.set_title('Acceleration')
            ax2.plot(ts, exp_data[1] - pred_data[1], label=[f'Mass {i}' for i in range(3)])
            ax2.set_xlabel('Time [$s$]')
            ax2.set_ylabel('Acceleration error [$m/s^2$]')
            # ax2.set_ylim([-0.3,0.3])
            ax2.legend()

            plt.tight_layout()
            fig.savefig(os.path.join(plot_dir, f'{prefix}_error.png'))
            plt.show() if show else plt.close()

            for i in range(3):
                fig, (ax1, ax2) = plt.subplots(2,1)
                fig.suptitle(f'{prefix}: Mass {i}')
                ax1.set_title(f'Position')
                ax1.plot(ts, pred_data[0,:,i], label='predicted')
                ax1.plot(ts, exp_data[0,:,i], label='expected')
                ax1.set_xlabel('Time [$s$]')
                ax1.set_ylabel('Position [$m$]')
                # ax1.set_ylim([-0.5,0.5])
                ax1.legend()

                ax2.set_title(f'Acceleration')
                ax2.plot(ts, pred_data[1,:,i], label='predicted')
                ax2.plot(ts, exp_data[1,:,i], label='expected')
                ax2.set_xlabel('Time [$s$]')
                ax2.set_ylabel('Acceleration [$m/s^2$]')
                # ax2.set_ylim([-0.3,0.3])
                ax2.legend()

                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'{prefix}_mass{i}.png'))
                if show: plt.show()
                plt.close()
        
        elif config.net_params.prediction == 'position':
            fig, ax1 = plt.subplots(1)

            ax1.set_title('Position')
            ax1.plot(ts, exp_data - pred_data, label=[f'Mass {i}' for i in range(3)])
            ax1.set_xlabel('Time [$s$]')
            ax1.set_ylabel('Position error [$m$]')
            # ax1.set_ylim([-0.5,0.5])
            ax1.legend()
            plt.tight_layout()
            fig.savefig(os.path.join(plot_dir, f'{prefix}_error.png'))
            plt.show() if show else plt.close()

            for i in range(3):
                fig, ax1 = plt.subplots(1)
                fig.suptitle(f'{prefix}: Mass {i}')
                ax1.set_title(f'Position')
                ax1.plot(ts, pred_data[:,i], label='predicted')
                ax1.plot(ts, exp_data[:,i], label='expected')
                ax1.set_xlabel('Time [$s$]')
                ax1.set_ylabel('Position [$m$]')
                # ax1.set_ylim([-0.5,0.5])
                ax1.legend()

                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'{prefix}_mass{i}.png'))
                if show: plt.show()
                plt.close()
        plt.close()

    ckpt = checkpoint.Checkpoint(checkpoint_dir)
    state = ckpt.restore_or_initialize(state)

    trajs_size = len(train_data)
    ts_size = len(train_data[0]) - config.training_params.horizon
    steps_per_epoch = ts_size // config.training_params.batch_size * trajs_size // config.training_params.batch_size

    # steps_per_epoch = ts_size // config.training_params.batch_size

    init_epoch = int(state.step) // steps_per_epoch + 1
    final_epoch = init_epoch + config.training_params.num_epochs
    config.training_params.num_epochs = final_epoch

    train_metrics = None
    print("Start training")
    for epoch in range(init_epoch, final_epoch):
        net.training = True
        rng, train_rng = jax.random.split(rng)
        state, metrics_update = train_epoch_1_traj(state, train_data, config.training_params.batch_size, train_rng) # test fit on one trajectory for now
        # state, metrics_update = train_epoch(state, train_data, config.training_params.batch_size, train_rng)
        if train_metrics is None:
            train_metrics = metrics_update
        else:
            train_metrics = train_metrics.merge(metrics_update)
        writer.write_scalars(epoch, add_prefix_to_keys(train_metrics.compute(), 'train'))
        report_progress(epoch)

        is_last_step = (epoch == final_epoch - 1)

        if epoch % config.training_params.eval_every_steps == 0 or is_last_step:
            with report_progress.timed('eval'):
                rng, eval_rng = jax.random.split(rng)
                rng_idx = jax.random.randint(eval_rng, [1], minval=0, maxval=len(eval_data)-1).item()
                # ts, pred_data, exp_data = rollout(state, eval_data, rng_idx)
                ts, pred_data, exp_data = rollout(state, train_data)
            plot(ts, pred_data, exp_data, prefix=f'Epoch {epoch}')
            
        if epoch % config.training_params.checkpoint_every_steps == 0 or is_last_step:
            with report_progress.timed('checkpoint'):
                ckpt.save(state)

        if epoch % config.training_params.clear_cache_every_steps == 0 or is_last_step: 
            jax.clear_caches()

        print(f'Epoch {epoch}: loss = {round(train_metrics.compute()["loss"], 4)}')

    config_js = config.to_json_best_effort()
    run_params_file = os.path.join(work_dir, 'run_params.js')
    with open(run_params_file, "w") as outfile:
        json.dump(config_js, outfile)

def minimal_working_example():
    rng = jax.random.key(0)
    rng, init_rng = jax.random.split(rng)

    batch_size = 10

    tx = optax.adam(1e-3)
    net = GraphNet(**config.net_params)
    init_graphs = build_graph(dataset_path=config.config.training_data_path,
                             key=init_rng,
                             batch_size=batch_size,
                             horizon=config.training_params.horizon,
                             render=False)
    params = net.init(init_rng, init_graphs[0])
    batched_apply = jax.vmap(net.apply, in_axes=(None,0))

    state = TrainState.create(
        apply_fn=batched_apply,
        params=params,
        tx=tx,
    )

    batched_graph = pytrees_stack(init_graphs)
    y = jnp.ones((batch_size,3,3)) # [batch_size, graph nodes, graph features]
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
    args = parser.parse_args()

    config = create_gnn_config(args)
    test_graph_network(config)
    # minimal_working_example()