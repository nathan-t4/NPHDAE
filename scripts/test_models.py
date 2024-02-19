import os
import optax
import matplotlib.pyplot as plt

from time import strftime
from functools import partial
from frozendict import frozendict
from clu import metric_writers
from clu import metrics
from clu import periodic_actions
from clu import checkpoint
from flax.training import train_state

from scripts.models import *
from scripts.data_utils import *
from utils.train_utils import *

training_data_path = 'results/double_mass_spring_data/no_control_train.pkl'
evaluation_data_path = 'results/double_mass_spring_data/no_control_val.pkl'

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50' # default is .75

def test_neural_ODE(args):
    """ Test Neural ODEs """

    data = load_data_jnp(training_data_path)

    if args.dir == None:
        work_dir = os.path.join(os.curdir, f'results/test_models/{strftime("%m%d")}_test_neural_ode/{strftime("%H%M%S")}')
    else:
        work_dir = args.dir

    log_dir = os.path.join(work_dir, 'log')
    checkpoint_dir = os.path.join(work_dir, 'checkpoint')
    plot_dir = os.path.join(work_dir, 'plots')

    rng = jax.random.key(0)
    rng, init_rng = jax.random.split(rng)

    # Create writer for logs
    writer = metric_writers.create_default_writer(logdir=log_dir)

    training_params = frozendict({
        'num_epochs': int(1),
        'traj_idx': 0,
        'horizon': 10,
        'batch_size': 5,
        'rollout_timesteps': 20,
        'checkpoint_every_steps': 10,
    })

    derivative_net_params = frozendict({
        'feature_sizes': [3,10,10,3], # [input, hidden, output]
        'activation': 'relu',
    })

    derivative_net = MLP(**derivative_net_params)

    net_params = frozendict({
        'derivative_net': derivative_net,
        'dt': 0.01,
    })

    net = NeuralODE(**net_params)
    init_input = jnp.ones(3)
    params = jax.jit(net.init)(init_rng, init_input)

    batched_apply = jax.vmap(net.apply, in_axes=(None,0,0))

    tx = optax.adam(learning_rate=int(1e-4))

    state = train_state.TrainState.create(
        apply_fn=batched_apply, params=params, tx=tx,
    )

    # Create logger to report training progress
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=training_params['num_epochs'], 
        writer=writer
    )

    ckpt = checkpoint.Checkpoint(checkpoint_dir)
    state = ckpt.restore_or_initialize(state)
    init_epoch = int(state.step)

    @jax.jit
    def loss_fn(params, data):
        qi = jnp.squeeze(data[:,0:3])
        qf = jnp.squeeze(data[:,3:6])
        t0 = jnp.squeeze(data[:,-2])
        tf = jnp.squeeze(data[:,-1])
        pred_qf = state.apply_fn(params, qi, jnp.column_stack((t0, tf)))
        loss = optax.l2_loss(predictions=jnp.squeeze(pred_qf), targets=qf).mean()
        return loss
    
    def train_epoch(ds, state, batch_size, rng):
        ds_size = len(ds) - training_params['horizon']
        steps_per_epoch = ds_size // batch_size
        perms = jax.random.permutation(rng, ds_size)
        perms = perms[:steps_per_epoch * batch_size]
        perms = perms.reshape((steps_per_epoch, batch_size))
        perms = jnp.asarray(perms)
        epoch_loss = []

        print(perms.shape)
        
        def train_batch(perms, state):
            for t0s in perms[:50]: # TODO: change to just perms (was too slow), maybe parallelize?
                t0s = jnp.reshape(t0s, (-1,1))
                tfs = jnp.reshape(t0s + training_params['horizon'], (-1,1))
                batch_qis = jnp.squeeze(ds[t0s, 0:3])
                batch_qfs = jnp.squeeze(ds[tfs, 0:3])
                batch_data = jnp.concatenate((batch_qis, batch_qfs, t0s, tfs), axis=1)
                loss, grads = jax.value_and_grad(loss_fn)(state.params, batch_data)
                state = state.apply_gradients(grads=grads)
                epoch_loss.append(loss)
            
            train_loss = np.mean(loss)
            return state, train_loss
        
        state, train_loss = jax.jit(train_batch)(perms, state)

        return state, TrainMetrics.single_from_model_output(loss=train_loss)
    
    @jax.jit
    def rollout(state, ds, t0=0):
        init_qs = ds[t0, 0:3]
        exp_qs_buffer = ds[t0:t0+training_params['rollout_timesteps']+1, 0:3]
        pred_qs_buffer = []
        pred_qs_buffer += [init_qs]

        next_qs = np.reshape(init_qs, (1,-1))
        for i in range(training_params['rollout_timesteps']):
            ts = np.reshape([i, i+1], (1,-1)) * net_params['dt']
            next_qs = state.apply_fn(state.params, next_qs, ts)
            next_qs = np.reshape(next_qs, (1,-1))
            pred_qs_buffer.append(list(np.squeeze(next_qs)))

        pred_qs_buffer = jnp.asarray(pred_qs_buffer)
        exp_qs_buffer = jnp.asarray(exp_qs_buffer)

        return pred_qs_buffer, exp_qs_buffer
    
    def plot(pred_qs_buffer, exp_qs_buffer, prefix=None):
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)

        plt.title(f'{prefix}: Error from rollout')
        plt.plot(np.arange(len(pred_qs_buffer)), exp_qs_buffer - pred_qs_buffer)
        plt.xlabel('Rollout timesteps [0.01s]')
        plt.ylabel('Position error')
        plt.savefig(os.path.join(plot_dir, f'{prefix}_pos_error.png'))
        plt.show()

        plt.title(f'{prefix}: Position of mass 1')
        plt.plot(np.arange(len(pred_qs_buffer[:,1])), pred_qs_buffer[:,1], label='predicted')
        plt.plot(np.arange(len(pred_qs_buffer[:,1])), exp_qs_buffer[:,1], label='expected')
        plt.xlabel('Rollout timesteps [0.01s]')
        plt.ylabel('Position')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'{prefix}_mass1_pos.png'))
        plt.show()

        plt.title(f'{prefix}: Position of mass 2')
        plt.plot(np.arange(len(pred_qs_buffer[:,2])), pred_qs_buffer[:,2], label='predicted')
        plt.plot(np.arange(len(pred_qs_buffer[:,2])), exp_qs_buffer[:,2], label='expected')
        plt.xlabel('Rollout timesteps [0.01s]')
        plt.ylabel('Position')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'{prefix}_mass2_pos.png'))
        plt.show()
    
    train_metrics = None
    ds = data[training_params['traj_idx']]
    print("Start training")
    for epoch in range(init_epoch, init_epoch + training_params['num_epochs']):
        rng, train_rng = jax.random.split(rng)
        state, metrics_update = train_epoch(ds, state, training_params['batch_size'], train_rng)

        if train_metrics is None:
            train_metrics = metrics_update
        else:
            train_metrics = train_metrics.merge(metrics_update)
        writer.write_scalars(epoch, add_prefix_to_keys(train_metrics.compute(), 'train'))
        report_progress(epoch)

        is_last_step = (epoch == init_epoch + training_params['num_epochs'] - 1)

        if epoch & training_params['checkpoint_every_steps'] == 0 or is_last_step:
            with report_progress.timed('checkpoint'):
                ckpt.save(state)

        print(f'Epoch {epoch}: loss = {round(train_metrics.compute()["loss"], 4)}')

    save_params(work_dir, training_params, None) # TODO: cannot json dump MLP

    print("Start validation")
    pred_qs, exp_qs = rollout(state, ds)
    plot(pred_qs, exp_qs, prefix=f'Epoch {init_epoch + training_params["num_epochs"]}')

def test_graph_network():
    # TODO
    pass

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default=None)
    args = parser.parse_args()

    test_neural_ODE(args)
