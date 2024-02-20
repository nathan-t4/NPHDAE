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
from scripts.data_utils import *
from utils.train_utils import *

training_data_path = 'results/double_mass_spring_data/no_control_train.pkl'
evaluation_data_path = 'results/double_mass_spring_data/no_control_val.pkl'

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50' # default is .75

def create_neural_ODE_config(args):
    training_params = ml_collections.ConfigDict({
        'num_epochs': int(200),
        'traj_idx': 0,
        'horizon': 10,
        'batch_size': 5,
        'rollout_timesteps': 1500,
        'eval_every_steps': 10,
        'checkpoint_every_steps': 10,
    })

    derivative_net_params = ml_collections.ConfigDict({
        'feature_sizes': [3,10,10,3], # [input, hidden, output]
        'activation': 'relu',
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

    data = load_data_jnp(training_data_path)

    if config.config.dir == None:
        work_dir = os.path.join(os.curdir, f'results/test_models/{strftime("%m%d")}_test_neural_ode/{strftime("%H%M%S")}')
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

    net_params = ml_collections.FrozenConfigDict({
        'derivative_net': derivative_net,
        'dt': 0.01, # TODO: make sure this is consistent with ts[1] - ts[0]
    })

    net = NeuralODE(**net_params)
    x = jnp.ones((1,3))
    variables = net.init(init_rng, x)
    tx = optax.adam(1e-3)

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
        ds_size = len(ds) - config.training_params.horizon
        steps_per_epoch = ds_size // batch_size
        perms = jax.random.permutation(rng, ds_size)
        perms = perms[:steps_per_epoch * batch_size]
        perms = perms.reshape((steps_per_epoch, batch_size))
        perms = jnp.asarray(perms)
        epoch_loss = []
        
        def loss_fn(params, data):
            qi = data[:,0:3]
            qf = data[:,3:6]
            t0 = jnp.squeeze(data[:,-2])
            tf = jnp.squeeze(data[:,-1])
            pred_qf = state.apply_fn({'params': params}, qi, jnp.column_stack((t0, tf)) * net_params['dt'])
            # pred_qf = [batch_dim, ts, ys]
            loss = int(1e6) * optax.l2_loss(predictions=pred_qf[:,-1], targets=jnp.squeeze(qf)).mean()
            return loss
            
        def train_batch(state, t0s):
            t0s = jnp.reshape(t0s, (-1,1))
            tfs = jnp.reshape(t0s + config.training_params.horizon, (-1,1))
            batch_qis = jnp.squeeze(ds[t0s, 0:3], 1)
            batch_qfs = jnp.squeeze(ds[tfs, 0:3], 1)
            batch_data = jnp.concatenate((batch_qis, batch_qfs, t0s, tfs), axis=1)
            loss, grads = jax.value_and_grad(loss_fn)(state.params, batch_data)
            state = state.apply_gradients(grads=grads)

            return state, loss
        
        state, epoch_loss = jax.lax.scan(train_batch, state, perms)
            
        train_loss = jnp.asarray(epoch_loss).mean()

        return state, TrainMetrics.single_from_model_output(loss=train_loss)

    def rollout(state, ds, t0=0):
        init_qs = np.reshape(ds[t0, 0:3], (1,-1))
        exp_qs_buffer = ds[t0:t0+config.training_params.rollout_timesteps+1, 0:3]
        pred_qs_buffer = []

        ts = np.arange(t0, t0+config.training_params.rollout_timesteps+1).reshape(1,-1) * net_params['dt']
        next_qs = state.apply_fn({'params': state.params}, init_qs, ts)
        pred_qs_buffer = jnp.asarray(next_qs).squeeze(axis=0)
        exp_qs_buffer = jnp.asarray(exp_qs_buffer)
        ts = ts.squeeze(axis=0)

        return ts, pred_qs_buffer, exp_qs_buffer
    
    def plot(ts, pred_qs_buffer, exp_qs_buffer, prefix=None, show=False):
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)

        plt.title(f'{prefix}: Error from rollout')
        plt.plot(ts, exp_qs_buffer - pred_qs_buffer)
        plt.xlabel('Time [s]')
        plt.ylabel('Position error')
        plt.savefig(os.path.join(plot_dir, f'{prefix}_pos_error.png'))
        plt.show() if show else plt.clf()

        plt.title(f'{prefix}: Position of mass 1')
        plt.plot(ts, pred_qs_buffer[:,1], label='predicted')
        plt.plot(ts, exp_qs_buffer[:,1], label='expected')
        plt.xlabel('Time [s]')
        plt.ylabel('Position')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'{prefix}_mass1_pos.png'))
        plt.show() if show else plt.clf()

        plt.title(f'{prefix}: Position of mass 2')
        plt.plot(ts, pred_qs_buffer[:,2], label='predicted')
        plt.plot(ts, exp_qs_buffer[:,2], label='expected')
        plt.xlabel('Time [s]')
        plt.ylabel('Position')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'{prefix}_mass2_pos.png'))
        plt.show() if show else plt.clf()

    
    train_metrics = None
    ds = data[config.training_params.traj_idx]
    # eval_ds = data[config.training_params.eval_traj_idx] # TODO

    ckpt = checkpoint.Checkpoint(checkpoint_dir)
    state = ckpt.restore_or_initialize(state)
    ds_size = len(ds) - config.training_params.horizon
    steps_per_epoch = ds_size // config.training_params.batch_size

    init_epoch = int(state.step) // steps_per_epoch + 1
    final_epoch = init_epoch + config.training_params.num_epochs

    print("Start training")
    for epoch in range(init_epoch, final_epoch):
        rng, train_rng = jax.random.split(rng)
        state, metrics_update = train_epoch(state, ds, config.training_params.batch_size, train_rng)

        if train_metrics is None:
            train_metrics = metrics_update
        else:
            train_metrics = train_metrics.merge(metrics_update)
        writer.write_scalars(epoch, add_prefix_to_keys(train_metrics.compute(), 'train'))
        report_progress(epoch)

        is_last_step = (epoch == final_epoch - 1)

        if epoch % config.training_params.eval_every_steps == 0 or is_last_step:
            with report_progress.timed('eval'):
                ts, pred_qs, exp_qs = rollout(state, ds)
            plot(ts, pred_qs, exp_qs, prefix=f'Epoch {epoch}')
            del ts, pred_qs, exp_qs
            
        if epoch % config.training_params.checkpoint_every_steps == 0 or is_last_step:
            with report_progress.timed('checkpoint'):
                ckpt.save(state)

        print(f'Epoch {epoch}: loss = {round(train_metrics.compute()["loss"], 4)}')

    config.to_json()

    print("Start validation")
    ts, pred_qs, exp_qs = rollout(state, ds)
    plot(ts, pred_qs, exp_qs, prefix=f'Epoch {final_epoch}')

def test_graph_network():
    # TODO
    pass

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default=None)
    args = parser.parse_args()

    config = create_neural_ODE_config(args)
    test_neural_ODE(config)
