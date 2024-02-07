import jax
import optax
import os
import json

import numpy as np
import matplotlib.pyplot as plt

from time import strftime
from argparse import ArgumentParser
from typing import Tuple, Optional

from absl import logging
from clu import metric_writers
from clu import metrics
from clu import checkpoint
from clu import periodic_actions
from flax.training import train_state

from scripts.graphs import *
from scripts.data_utils import *
from scripts.models import *
from utils.logging_utils import *

training_data_path = 'results/double_mass_spring_data/no_control_train.pkl'
# validation_data_path = 'results/double_mass_spring_data/5uniform.pkl'
validation_data_path = 'results/double_mass_spring_data/no_control_val.pkl'

def train(args):
    platform = jax.local_devices()[0].platform
    print('Running on platform:', platform.upper())

    # work_dir = os.path.join(os.curdir, f'results/gnn/{strftime("%Y%m%d-%H%M%S")}')
    work_dir = os.path.join(os.curdir, f'results/gnn/{strftime("%m%d")}_generalization_control/passive_to_passive_{strftime("%H%M%S")}')
    log_dir = os.path.join(work_dir, 'log')
    checkpoint_dir = os.path.join(work_dir, 'checkpoints')

    model_type = 'GraphNet'

    net_params = {
        'num_message_passing_steps': 2,
        'use_edge_model': True,
    }

    training_params = {
        'batch_size': 5,
        'lr': 1e-4,
        'eval_every_steps': 20,
        'log_every_steps': 10,
        'checkpoint_every_steps': int(1e3),
        'num_train_steps': int(1e4),
    }

    def create_model(model_type: str, params):
        if model_type == 'GNODE':
            return GNODE(**params)
        elif model_type == 'GraphNet':
            return GraphNet(**params)
        else:
            raise NotImplementedError
    
    # Create key and initialize params
    rng = jax.random.key(0)
    rng, init_rng = jax.random.split(rng)

    # Create writer for logs
    writer = metric_writers.create_default_writer(logdir=log_dir)

    # Create the optimizer
    tx = optax.adam(learning_rate=training_params['lr'])

    # Create the training state
    net = create_model(model_type, net_params)
    init_graph = build_graph(dataset_path=args.training_data, key=init_rng, batch_size=1, render=False)[0]
    params = jax.jit(net.init)(init_rng, init_graph)
    # params = net.init(init_rng, init_graph)
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
        num_train_steps=training_params['num_train_steps'], 
        writer=writer
    )

    print("Starting training")
    train_metrics = None
    # Training loop
    for epoch in range(init_epoch, init_epoch + training_params['num_train_steps']):
        rng, dropout_rng, graph_rng = jax.random.split(rng, 3)
    
        with jax.profiler.StepTraceAnnotation('train', step_num=epoch):
            graphs = build_graph(dataset_path=args.training_data,
                                 key=graph_rng,
                                 batch_size=training_params['batch_size'])
            state, metrics_update = train_step(state, graphs, rngs={'dropout': dropout_rng})
            # Update metrics
            if train_metrics is None:
                train_metrics = metrics_update
            else:
                train_metrics = train_metrics.merge(metrics_update)
        
        print(f"Training step {epoch} - training loss {round(train_metrics.compute()['loss'], 4)}")

        report_progress(epoch)

        is_last_step = (epoch == training_params['num_train_steps'] - 1)

        if epoch % training_params['log_every_steps'] == 0 or is_last_step:
            writer.write_scalars(epoch, add_prefix_to_keys(train_metrics.compute(), 'train'))
            train_metrics = None

        if epoch % training_params['eval_every_steps'] == 0 or is_last_step:
            eval_state = eval_state.replace(params=state.params)
            rng, eval_rng = jax.random.split(rng)
            eval_graphs = build_graph(dataset_path=args.validation_data, 
                                      key=eval_rng,
                                      batch_size=1)
            with report_progress.timed('eval'):
                eval_metrics = eval_model(eval_state, eval_graphs)
                writer.write_scalars(epoch, add_prefix_to_keys(eval_metrics.compute(), 'eval'))

        if epoch & training_params['checkpoint_every_steps'] == 0 or is_last_step:
            with report_progress.timed('checkpoint'):
                ckpt.save(state)
    
    # Validation loop
    print("Validating policy")
    val_state = eval_state.replace(params=state.params)
    rng, val_rng = jax.random.split(rng)
    val_graphs = build_graph(dataset_path=args.validation_data,
                             key=val_rng,
                             batch_size=10)
    with report_progress.timed('val'):
        val_metrics = eval_model(val_state, val_graphs)
        writer.write_scalars(init_epoch + training_params['num_train_steps'], 
                             add_prefix_to_keys(val_metrics.compute(), 'val'))

    print(f"Validation loss {round(val_metrics.compute()['loss'],4)}")

    # Save run params to json
    run_params = {
        'training_params': training_params,
        'net_params': net_params
    }
    run_params_file = os.path.join(work_dir, 'run_params.js')
    with open(run_params_file, "w") as outfile:
        json.dump(run_params, outfile)

def get_labelled_data(graphs: Sequence[jraph.GraphsTuple], 
                      traj_idx: int = 0) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    data = load_data_jnp(path=training_data_path)[traj_idx]

    qs = []
    dqs = []
    ps = []
    vs = []
    accs = []

    for g in graphs:
        t = g.globals[0]
        masses = (g.globals[1:4]).squeeze()
        # expected x, dx, and p at time t
        qs += [data[t,0:3]]
        dqs += [data[t,3:5]]
        ps += [data[t,5:8]]
        vs += [data[t,5:8] / masses] # component wise division
        accs += [data[t,8:11]]

    return jnp.array(qs), jnp.array(dqs), jnp.array(ps), jnp.array(vs), jnp.array(accs)

@jax.jit
def mse_loss_fn(expected, predicted):
    return jnp.sum(optax.l2_loss(expected - predicted))

@jax.jit
def train_step(state: train_state.TrainState, 
               graphs: Sequence[jraph.GraphsTuple],
               rngs: Dict[str, jnp.ndarray]) -> Tuple[train_state.TrainState, metrics.Collection]:
    def loss_fn(params, graphs: Sequence[jraph.GraphsTuple]):
        curr_state = state.replace(params=params)
        pred_as = []
        for g in graphs:
            pred_graph = curr_state.apply_fn(curr_state.params, g, rngs=rngs)
            pred_as += [pred_graph.nodes.reshape(-1)]
        
        _, _, _, _, expected_as = get_labelled_data(graphs)

        loss = 0
        for i in range(len(graphs)):
            '''For acceleration predictions'''
            predicted_as = jnp.asarray(pred_as[i])
            loss += mse_loss_fn(expected_as[i], predicted_as)

        return loss / len(graphs)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False, allow_int=True)
    loss, grads = grad_fn(state.params, graphs)
    # Apply gradient to optimizer
    state = state.apply_gradients(grads=grads)
    # Update metrics
    metrics = TrainMetrics.single_from_model_output(loss=loss)

    return state, metrics

@jax.jit
def eval_step(state: train_state.TrainState,
              graphs: Sequence[jraph.GraphsTuple]) -> metrics.Collection:
    curr_state = state.replace(params=state.params)
    pred_as = []

    for g in graphs:
        pred_graph = curr_state.apply_fn(state.params, g)
        pred_as += [pred_graph.nodes.reshape(-1)]

    _, _, _, _, expected_as = get_labelled_data(graphs)
    
    loss = 0
    for i in range(len(graphs)):
        '''For acceleration predictions'''
        predicted_as = jnp.asarray(pred_as[i])
        loss += mse_loss_fn(expected_as[i], predicted_as)
    
    loss = loss / len(graphs)
    
    return EvalMetrics.single_from_model_output(loss=loss)

@jax.jit
def eval_model(state: train_state.TrainState, 
               graphs: Sequence[jraph.GraphsTuple]) -> metrics.Collection:
    eval_metrics = None

    metrics_update = eval_step(state, graphs)

    if eval_metrics is None:
        eval_metrics = metrics_update
    else:
        eval_metrics = eval_metrics.merge(metrics_update)

    return eval_metrics

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--training_data', type=str, default=training_data_path)
    parser.add_argument('--validation_data', type=str, default=validation_data_path)
    args = parser.parse_args()

    train(args)