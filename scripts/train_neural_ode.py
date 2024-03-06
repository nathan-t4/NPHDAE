import os
from time import strftime

import matplotlib.pyplot as plt

import optax
import ml_collections
from clu import metric_writers
from clu import periodic_actions
from clu import checkpoint
from flax.training.train_state import TrainState

from scripts.models import *
from utils.data_utils import *
from utils.train_utils import *

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50' # default is .75
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'

def create_neural_ODE_config(args):
    training_params = ml_collections.ConfigDict({
        'num_epochs': int(8e2),
        'traj_idx': 0,
        'horizon': 20, # VARY! 
        'batch_size': 8,
        'rollout_timesteps': 1500,
        'eval_every_steps': 50,
        'checkpoint_every_steps': 50,
        'clear_cache_every_steps': 50,
        'dt': 0.01, # Note: dt is arbitrary since horizon is fixed
    })

    derivative_net_params = ml_collections.ConfigDict({
        'feature_sizes': [16, 16, 3], # [hidden, output], inputs = (state, time)
        'activation': 'softplus',
        'deterministic': False,
    })

    config = ml_collections.ConfigDict()
    config.config = ml_collections.ConfigDict({
        'dir': args.dir,
        'training_data_path': 'results/double_mass_spring_data/no_control_train.pkl',
        'evaluation_data_path': 'results/double_mass_spring_data/no_control_val.pkl',

    })
    config.training_params = training_params
    config.derivative_net_params = derivative_net_params
    config.lock()

    return config

def test_neural_ODE(config: ml_collections.ConfigDict):
    """ Test Neural ODEs """
    if config.config.dir == None:
        work_dir = os.path.join(os.curdir, f'results/test_models/{strftime("%m%d")}_test_neural_ode/16_16_{strftime("%H%M%S")}')
    else:
        work_dir = config.config.dir

    log_dir = os.path.join(work_dir, 'log')
    checkpoint_dir = os.path.join(work_dir, 'checkpoint')
    plot_dir = os.path.join(work_dir, 'plots')

    rng = jax.random.key(0)
    rng, init_rng = jax.random.split(rng)

    # Create writer for logs
    writer = metric_writers.create_default_writer(logdir=log_dir)

    derivative_net = MLP(**config.derivative_net_params)

    net = NeuralODE(derivative_net=derivative_net)
    x = jnp.ones(3)
    variables = net.init(init_rng, x)
    # tx = optax.adam(1e-3)
    tx = optax.adabelief(3e-3)

    batched_apply = jax.vmap(net.apply, in_axes=(None,0,0))


    state = TrainState.create(
         apply_fn=batched_apply,
         params=variables['params'],
         tx=tx)
    
    # Create logger to report training progress
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=config.training_params.num_epochs,
        writer=writer
    )
    
    def train_epoch(state, ds, batch_size, rng):
        ds_size = len(ds)
        steps_per_epoch = ds_size // batch_size
        perms = jax.random.permutation(rng, ds_size)
        perms = perms[:steps_per_epoch * batch_size]
        perms = perms.reshape((steps_per_epoch, batch_size))
        perms = jnp.asarray(perms)
        epoch_loss = []
        
        def loss_fn(params, data):
            qi = data[:,0:3]
            qf = data[:,3:6]
            t0 = jnp.squeeze(data[:,-2]) * config.training_params['dt']
            tf = jnp.squeeze(data[:,-1]) * config.training_params['dt']
            pred_qf = state.apply_fn({'params': params}, qi, jnp.column_stack((t0, tf)))
            # pred_qf = [batch_dim, ts, ys] 
            loss = int(1e6) * optax.l2_loss(predictions=pred_qf[:,-1,:], targets=qf).mean() # get pred_qf at last timestep only
            return loss
            
        def train_batch(state, t0s):
            t0s = jnp.reshape(t0s, (-1,1)) 
            tfs = jnp.reshape(t0s + config.training_params.horizon, (-1,1)).clip(max=ds_size)
            batch_qis = jnp.squeeze(ds[t0s, 0:3], 1)
            batch_qfs = jnp.squeeze(ds[tfs, 0:3], 1)
            batch_data = jnp.concatenate((batch_qis, batch_qfs, t0s, tfs), axis=1)
            loss, grads = jax.value_and_grad(loss_fn)(state.params, batch_data)

            state = state.apply_gradients(grads=grads)

            return state, loss
        
        state, epoch_loss = jax.lax.scan(train_batch, state, perms)
        train_loss = jnp.asarray(epoch_loss).mean()

        return state, TrainMetrics.single_from_model_output(loss=train_loss)

    def rollout(state, ds, t0=0, dt=config.training_params['dt']):
        init_qs = ds[t0, 0:3]
        exp_qs_buffer = ds[t0:t0+config.training_params.rollout_timesteps+1, 0:3]
        inputs = np.reshape(init_qs, (1,-1))
        ts = np.arange(t0, t0+config.training_params.rollout_timesteps+1).reshape(1,-1) * dt
        next_qs = state.apply_fn({'params': state.params}, inputs, ts)
        pred_qs_buffer = jnp.asarray(next_qs).squeeze(axis=0)
        exp_qs_buffer = jnp.asarray(exp_qs_buffer)
        ts = ts.squeeze(axis=0)

        return ts, pred_qs_buffer, exp_qs_buffer
    
    def plot(ts, pred_qs_buffer, exp_qs_buffer, prefix=None, show=False):
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)

        plt.title(f'{prefix}: Error from rollout')
        plt.plot(ts, exp_qs_buffer - pred_qs_buffer, label=[f'Mass {i}' for i in range(3)])
        plt.xlabel('Time [s]')
        plt.ylabel('Position error')
        # plt.ylim((-0.5,0.5))
        plt.savefig(os.path.join(plot_dir, f'{prefix}_pos_error.png'))
        plt.show() if show else plt.clf()

        plt.title(f'{prefix}: Position of mass 0 (ground)')
        plt.plot(ts, pred_qs_buffer[:,0], label='predicted')
        plt.plot(ts, exp_qs_buffer[:,0], label='expected')
        plt.xlabel('Time [s]')
        plt.ylabel('Position')
        # plt.ylim((-0.5,0.5))
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'{prefix}_mass0_pos.png'))
        plt.show() if show else plt.clf()

        plt.title(f'{prefix}: Position of mass 1')
        plt.plot(ts, pred_qs_buffer[:,1], label='predicted')
        plt.plot(ts, exp_qs_buffer[:,1], label='expected')
        plt.xlabel('Time [s]')
        plt.ylabel('Position')
        # plt.ylim((-0.5,0.5))
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'{prefix}_mass1_pos.png'))
        plt.show() if show else plt.clf()

        plt.title(f'{prefix}: Position of mass 2')
        plt.plot(ts, pred_qs_buffer[:,2], label='predicted')
        plt.plot(ts, exp_qs_buffer[:,2], label='expected')
        plt.xlabel('Time [s]')
        plt.ylabel('Position')
        # plt.ylim((-0.5,0.5))
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'{prefix}_mass2_pos.png'))
        plt.show() if show else plt.clf()

    train_data, _ = load_data_jnp(config.config.training_data_path)
    eval_data, _ = load_data_jnp(config.config.evaluation_data_path)
    
    ckpt = checkpoint.Checkpoint(checkpoint_dir)
    state = ckpt.restore_or_initialize(state)

    num_train_trajs = np.shape(train_data)[0]
    num_eval_trajs = np.shape(eval_data)[0]

    init_ds = train_data[0]
    ds_size = len(init_ds) - config.training_params.horizon
    steps_per_epoch = ds_size // config.training_params.batch_size

    init_epoch = int(state.step) // steps_per_epoch + 1
    final_epoch = init_epoch + config.training_params.num_epochs

    train_metrics = None
    print("Start training")
    for epoch in range(init_epoch, final_epoch):
        rng, train_rng = jax.random.split(rng)
        train_ds = train_data[0]
        state, metrics_update = train_epoch(state, train_ds, config.training_params.batch_size, train_rng)

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
                rng_idx = jax.random.randint(eval_rng, [1], minval=0, maxval=num_eval_trajs-1).item()
                # eval_ds = eval_data[rng_idx]
                eval_ds = train_data[0]
                ts, pred_qs, exp_qs = rollout(state, eval_ds)
            plot(ts, pred_qs, exp_qs, prefix=f'Epoch {epoch}')
            
        if epoch % config.training_params.checkpoint_every_steps == 0 or is_last_step:
            with report_progress.timed('checkpoint'):
                ckpt.save(state)

        if epoch % config.training_params.clear_cache_every_steps == 0 or is_last_step: 
            jax.clear_caches()

        print(f'Epoch {epoch}: loss = {round(train_metrics.compute()["loss"], 4)}')

    config_js = config.to_json()
    run_params_file = os.path.join(work_dir, 'run_params.js')
    with open(run_params_file, "w") as outfile:
        json.dump(config_js, outfile)

    print("Start validation")
    rng, val_rng = jax.random.split(rng)
    rng_idx = jax.random.randint(val_rng, [1], minval=0, maxval=num_eval_trajs-1).item()
    # val_ds = eval_data[rng_id# dt is arbitrary since horizon is fixed...x]
    val_ds = train_data[0]
    ts, pred_qs, exp_qs = rollout(state, val_ds)
    plot(ts, pred_qs, exp_qs, prefix=f'Epoch {final_epoch}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default=None)
    args = parser.parse_args()

    config = create_neural_ODE_config(args)
    test_neural_ODE(config)
