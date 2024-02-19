import jax
import optax
import os
import json

from time import strftime
from argparse import ArgumentParser
from typing import Tuple
from frozendict import frozendict

from clu import metric_writers
from clu import metrics
from clu import checkpoint
from clu import periodic_actions
from flax.training import train_state

from scripts.graphs import *
from scripts.data_utils import *
from scripts.models import *
from utils.train_utils import *

# Prevent GPU out-of-memory (OOM) errors 
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50' # default is .75

training_data_path = 'results/double_mass_spring_data/no_control_train.pkl'
validation_data_path = 'results/double_mass_spring_data/no_control_val.pkl'

# validation_data_path = 'results/double_mass_spring_data/5uniform.pkl'

# training_data_path = 'results/switched_double_mass_spring_data/1_switch_passive_train.pkl'
# validation_data_path = 'results/switched_double_mass_spring_data/1_switch_passive_val.pkl'

def train(args):
    """ Train and evaluate network """
    platform = jax.local_devices()[0].platform
    print('Running on platform:', platform.upper())

    # work_dir = os.path.join(os.curdir, f'results/gnn/{strftime("%Y%m%d-%H%M%S")}')
    work_dir = os.path.join(os.curdir, f'results/gnn/{strftime("%m%d")}_test_models/GNN_10_mp_steps_1_traj{strftime("%H%M%S")}')
    log_dir = os.path.join(work_dir, 'log')
    plot_dir = os.path.join(work_dir, 'plots')
    checkpoint_dir = os.path.join(work_dir, 'checkpoints')

    global net_params, training_params

    # Network parameters
    model_type = 'GraphNet'
    net_params = frozendict({
        'num_message_passing_steps': 1,
        'use_edge_model': True,
        'dropout_rate': 0.1,
    })
    # Training parameters
    training_params = frozendict({
        'batch_size': 5,
        'horizon': 5,
        'lr': 1e-4,
        'train_steps': 2,
        'eval_steps': 5,
        'eval_every_steps': 50,
        'log_every_steps': 20,
        'checkpoint_every_steps': int(1e3),
        'plot_every_steps': int(1e2),
        'num_epochs': int(5e3),
        'training_dataset_path': training_data_path,
        'validation_dataset_path': validation_data_path,
    })

    def create_model(model_type: str, params):
        if model_type == 'GNODE':
            return GNODE(**params)
        elif model_type == 'GraphNet':
            return GraphNet(**params)
        elif model_type == 'NeuralODE':
            return NeuralODE(**params) # TODO!!!!
        else:
            raise NotImplementedError(f'Invalid model type {model_type}')
    
    # Create key and initialize params
    rng = jax.random.key(0)
    rng, init_rng = jax.random.split(rng)

    # Create writer for logs
    writer = metric_writers.create_default_writer(logdir=log_dir)

    # Create the optimizer
    tx = optax.adam(learning_rate=training_params['lr'])

    # Create the training state
    net = create_model(model_type, net_params)
    init_graph = build_graph(dataset_path=args.training_data, 
                             key=init_rng, 
                             batch_size=1, 
                             horizon=training_params['horizon'], 
                             render=False)[0]
    # params = jax.jit(net.init)(init_rng, init_graph)
    params = net.init(init_rng, init_graph)
    state = train_state.TrainState.create(
        apply_fn=net.apply, params=params, tx=tx,
    )

    # Setup checkpointing for model
    ckpt = checkpoint.Checkpoint(checkpoint_dir)
    state = ckpt.restore_or_initialize(state)
    init_epoch = int(state.step) + 1

    # Create the evaluation net
    eval_net = create_model(model_type, net_params)
    eval_state = state.replace(apply_fn=eval_net.apply)

    # Create logger to report training progress
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=training_params['num_epochs'], 
        writer=writer
    )

    print("Starting training")
    train_metrics = None
    # Training loop
    for epoch in range(init_epoch, init_epoch + training_params['num_epochs']):
        rng, dropout_rng, graph_rng = jax.random.split(rng, 3)
    
        with jax.profiler.StepTraceAnnotation('train', step_num=epoch):
            graphs = build_graph(dataset_path=args.training_data,
                                 key=graph_rng,
                                 batch_size=training_params['batch_size'],
                                 horizon=training_params['horizon'])
            state, metrics_update = train_step(state, graphs, rngs={'dropout': dropout_rng})
            # Update metrics
            if train_metrics is None:
                train_metrics = metrics_update
            else:
                train_metrics = train_metrics.merge(metrics_update)
        
        print(f"Training step {epoch} - training loss {round(train_metrics.compute()['loss'], 4)}")

        report_progress(epoch)

        is_last_step = (epoch == training_params['num_epochs'])

        if epoch % training_params['log_every_steps'] == 0 or is_last_step:
            writer.write_scalars(epoch, add_prefix_to_keys(train_metrics.compute(), 'train'))
            train_metrics = None

        if epoch % training_params['eval_every_steps'] == 0 or is_last_step:
            eval_state = eval_state.replace(params=state.params)
            rng, eval_rng = jax.random.split(rng)
            eval_graphs = build_graph(dataset_path=args.validation_data, 
                                      key=eval_rng,
                                      batch_size=1,
                                      horizon=training_params['horizon'])
            with report_progress.timed('eval'):
                eval_metrics, pred_qs, exp_qs, pred_as, exp_as = eval_model(eval_state, eval_graphs)
                writer.write_scalars(epoch, add_prefix_to_keys(eval_metrics.compute(), 'eval'))
                if epoch % training_params['plot_every_steps'] == 0 or is_last_step:
                    save_evaluation_curves(plot_dir, f'{epoch}_position_eval', pred_qs, exp_qs)
                    save_evaluation_curves(plot_dir, f'{epoch}_acceleration_eval', pred_as, exp_as)
                    if (eval_metrics.compute()['loss'] < int(1e-6)):
                        print(f"Stop training early since evaluation error is {eval_metrics.compute()['loss']}")
                        training_params['num_epochs'] = epoch - init_epoch
                        ckpt.save(state)
                        break

        if epoch & training_params['checkpoint_every_steps'] == 0 or is_last_step:
            with report_progress.timed('checkpoint'):
                ckpt.save(state)

    save_params(work_dir, training_params, net_params)
    
    # Validation loop
    print("Validating policy")
    val_state = eval_state.replace(params=state.params)
    rng, val_rng = jax.random.split(rng)
    val_graphs = build_graph(dataset_path=args.validation_data,
                             key=val_rng,
                             batch_size=2,
                             horizon=training_params['horizon'])
    with report_progress.timed('val'):
        val_metrics, pred_qs, exp_qs, pred_as, exp_as = eval_model(val_state, val_graphs)
        save_evaluation_curves(plot_dir, 'position_val', pred_qs, exp_qs)
        save_evaluation_curves(plot_dir, 'acceleration_val', pred_as, exp_as)
        writer.write_scalars(init_epoch + training_params['num_epochs'], 
                             add_prefix_to_keys(val_metrics.compute(), 'val'))

    print(f"Validation loss {round(val_metrics.compute()['loss'],4)}")

@jax.jit
def get_labelled_data(graphs: Sequence[jraph.GraphsTuple], 
                      traj_idxs: Sequence[int] = [0]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    
    if isinstance(traj_idxs, int):
        traj_idxs = [traj_idxs]
    
    data = load_data_jnp(path=training_data_path) # TODO: pad with zeros?

    relevant_data = []
    for i in traj_idxs:
        relevant_data += [data[i]]
    
    relevant_data = jnp.array(relevant_data).squeeze(axis=1)
    
    qs = []
    dqs = []
    ps = []
    vs = []
    accs = []

    for traj_idx, g in enumerate(graphs):
        t = g.globals[1]
        masses = (g.globals[3:6]).squeeze()
        # expected x, dx, and p at time t
        qs += [relevant_data[traj_idx,t,0:3]]
        dqs += [relevant_data[traj_idx,t,3:5]]
        ps += [relevant_data[traj_idx,t,5:8]]
        vs += [relevant_data[traj_idx,t,5:8] / masses] # component wise division TODO: fix error
        accs += [relevant_data[traj_idx,t,8:11]]

    return jnp.array(qs), jnp.array(dqs), jnp.array(ps), jnp.array(vs), jnp.array(accs)

@jax.jit
def mse_loss_fn(expected, predicted):
    return jnp.mean(optax.l2_loss(expected - predicted))

@jax.jit
def train_step(state: train_state.TrainState, 
               graphs: Sequence[jraph.GraphsTuple],
               rngs: Dict[str, jnp.ndarray]) -> Tuple[train_state.TrainState, metrics.Collection]:
    def loss_fn(params, graphs: Sequence[jraph.GraphsTuple]):
        curr_state = state.replace(params=params)
        pred_as = []; exp_as = []
        loss = 0
        
        for i, g in enumerate(graphs): # TODO: can use jax.vmap
            for _ in jnp.arange(training_params['train_steps']):
                traj_idx = g.globals[0]
                _, _, _, _, exp_a = get_labelled_data([g], [traj_idx])
                g = curr_state.apply_fn(curr_state.params, g, rngs=rngs)
                pred_as += [g.nodes[:,-1].reshape(-1)]
                exp_as += [exp_a.reshape(-1)]
                g = g._replace(nodes=g.nodes[:,:-1]) # remove acceleration, keep only pos and vels
        
        exp_as = jnp.asarray(exp_as)
        pred_as = jnp.asarray(pred_as)
        loss += mse_loss_fn(exp_as, pred_as)

        return loss / len(graphs)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(state.params, graphs)
    state = state.apply_gradients(grads=grads)
    # Update metrics
    metrics = TrainMetrics.single_from_model_output(loss=loss)

    return state, metrics

@jax.jit
def eval_step(state: train_state.TrainState,
              graphs: Sequence[jraph.GraphsTuple]) -> metrics.Collection:
    curr_state = state.replace(params=state.params)
    pred_as = []
    traj_idxs = []

    for g in graphs:
        traj_idxs += [g.globals[0]]
        pred_graph = curr_state.apply_fn(curr_state.params, g)
        pred_as += [pred_graph.nodes[:,-1].reshape(-1)]

    _, _, _, _, expected_as = get_labelled_data(graphs, traj_idxs)
    
    loss = 0
    for i in range(len(graphs)):
        '''For acceleration predictions'''
        predicted_as = jnp.asarray(pred_as[i])
        loss += mse_loss_fn(expected_as[i], predicted_as)
    
    loss = loss / len(graphs)
    
    return EvalMetrics.single_from_model_output(loss=loss)

@jax.jit
def eval_traj(state: train_state.TrainState,
              graphs: Sequence[jraph.GraphsTuple]) -> Tuple[metrics.Collection, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """ Evaluate GNN performance for multiple timesteps - generalizes eval_step to multiple steps """
    
    curr_state = state.replace(params=state.params)
    pred_as = []; exp_as = []
    pred_qs = []; exp_qs = []
    loss = 0   
    for g in graphs:
        for _ in range(training_params['eval_steps']):
            traj_idx = g.globals[0]
            g = curr_state.apply_fn(curr_state.params, g)
            exp_q, _, _, _, exp_a = get_labelled_data([g], [traj_idx])
            pred_qs += [g.nodes[:,0].reshape(-1)]
            exp_qs += [exp_q.reshape(-1)]
            pred_as += [g.nodes[:,-1].reshape(-1)]
            exp_as += [exp_a.reshape(-1)]
            g = g._replace(nodes=g.nodes[:,:-1]) # remove acceleration, keep only next pos and vel

    exp_qs = jnp.asarray(exp_qs)
    pred_qs = jnp.asarray(pred_qs)
    exp_as = jnp.asarray(exp_as)
    pred_as = jnp.asarray(pred_as)
    loss += mse_loss_fn(exp_as, pred_as)

    loss = loss / len(graphs)

    return EvalMetrics.single_from_model_output(loss=loss), pred_qs, exp_qs, pred_as, exp_as

@jax.jit
def eval_model(state: train_state.TrainState, 
               graphs: Sequence[jraph.GraphsTuple]) -> metrics.Collection:
    """ Evaluate network using eval_traj """
    eval_metrics = None

    # metrics_update = eval_step(state, graphs)
    metrics_update, pred_qs, exp_qs, pred_as, exp_as = eval_traj(state, graphs)

    if eval_metrics is None:
        eval_metrics = metrics_update
    else:
        eval_metrics = eval_metrics.merge(metrics_update)

    return eval_metrics, pred_qs, exp_qs, pred_as, exp_as

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--training_data', type=str, default=training_data_path)
    parser.add_argument('--validation_data', type=str, default=validation_data_path)
    args = parser.parse_args()

    train(args)