import jax
import optax
import os

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

default_dataset_path = 'results/double_mass_spring_data/Double_Spring_Mass_2023-12-26-14-29-05.pkl'

def train(args):
    platform = jax.local_devices()[0].platform
    print('Running on platform:', platform.upper())

    # work_dir = os.path.join(os.curdir, f'results/gnn/{strftime("%Y%m%d-%H%M%S")}')
    work_dir = os.path.join(os.curdir, f'results/gnn/{strftime("%m%d")}_features_study/features_0_{strftime("%H%M%S")}')
    log_dir = os.path.join(work_dir, 'log')
    checkpoint_dir = os.path.join(work_dir, 'checkpoints')

    net_params = {
        'num_message_passing_steps': 2, # TODO: ablation study
        'use_edge_model': True, # DONE: ablation study
    }

    training_params = {
        'batch_size': 5,
        'lr': 1e-4,
        'eval_every_steps': 20,
        'log_every_steps': 10,
        'checkpoint_every_steps': int(1e3),
        'num_train_steps': int(1e4),
    }

    # Create key and initialize params
    rng = jax.random.key(0)
    rng, init_rng = jax.random.split(rng)

    # Create writer for logs
    writer = metric_writers.create_default_writer(logdir=log_dir)

    # Create the optimizer
    tx = optax.adam(learning_rate=training_params['lr'])

    # Create the training state
    net = GraphNet(**net_params)
    init_graph = build_graph(path=args.data, key=init_rng, batch_size=1, render=False)[0]
    params = jax.jit(net.init)(init_rng, init_graph)
    state = train_state.TrainState.create(
        apply_fn=net.apply, params=params, tx=tx,
    )

    # Setup checkpointing for model
    ckpt = checkpoint.Checkpoint(checkpoint_dir)
    state = ckpt.restore_or_initialize(state)
    init_epoch = int(state.step) + 1

    # Create the evaluation net
    eval_net = GraphNet(**net_params)
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
            graphs = build_graph(path=args.data,
                                 key=graph_rng,
                                 dataset_type='training', 
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
            eval_graphs = build_graph(path=args.data, 
                                      key=eval_rng,
                                      dataset_type='validation',
                                      batch_size=1)
            with report_progress.timed('eval'):
                eval_metrics = eval_model(eval_state, eval_graphs)
                writer.write_scalars(epoch, add_prefix_to_keys(eval_metrics.compute(), 'eval'))

        if epoch & training_params['checkpoint_every_steps'] == 0 or is_last_step:
            with report_progress.timed('checkpoint'):
                ckpt.save(state)
    
    # validation loop
    val_state = eval_state.replace(params=state.params)
    rng, val_rng = jax.random.split(rng)
    val_graphs = build_graph(path=args.data,
                             key=val_rng,
                             dataset_type='validation',
                             batch_size=10)
    with report_progress.timed('val'):
        val_metrics = eval_model(val_state, val_graphs)
        writer.write_scalars(init_epoch + training_params['num_train_steps'], 
                             add_prefix_to_keys(val_metrics.compute(), 'val'))

    print(f"Validation loss {round(val_metrics.compute()['loss'],4)}")

def get_labelled_data(graphs: Sequence[jraph.GraphsTuple], 
                      traj_idx: int = 0) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    data = load_data_jnp(path=default_dataset_path)[traj_idx]

    qs = []
    dqs = []
    ps = []
    vs = []

    for g in graphs:
        t = g.globals[0]
        masses = (g.globals[1:4]).squeeze()
        # expected x, dx, and p at time t
        qs += [data[t,0:3]]
        dqs += [data[t,3:5]]
        ps += [data[t,5:8]]
        vs += [data[t,5:8] / masses] # component wise division

    # exclude training for wall node?
    # pred_vs = pred_vs[1,:]
    # expected_vs = expected_vs[1:]
        
    # assert jnp.shape(vs[0]) == jnp.shape(ps[0]), \
    #     f"Error with component wise division using jax.numpy.true_divide: vs shape is {jnp.shape(vs[0])} while ps shape is {jnp.shape(ps[0])}"

    return jnp.array(qs), jnp.array(dqs), jnp.array(ps), jnp.array(vs)     

@jax.jit
def mse_loss_fn(expected, predicted):
    return jnp.sum(optax.l2_loss(expected - predicted))

@jax.jit
def train_step(state: train_state.TrainState, 
               graphs: Sequence[jraph.GraphsTuple],
               rngs: Dict[str, jnp.ndarray]) -> Tuple[train_state.TrainState, metrics.Collection]:
    # TODO: try predicting acceleration, then replacing node features 
    # calculate new node features (pos and vel) from acceleration and then graph._replace
    def loss_fn(params, graphs: Sequence[jraph.GraphsTuple]):
        curr_state = state.replace(params=params)
        pred_vs = []
        pred_dqs = []
        for g in graphs:
            pred_graph = curr_state.apply_fn(curr_state.params, g, rngs=rngs)
            pred_vs += [pred_graph.nodes.reshape(-1)]
            pred_dqs += [pred_graph.edges.reshape(-1)]
        
        _, expected_dqs, _, expected_vs = get_labelled_data(graphs)

        loss = 0
        for i in range(len(graphs)):
            predicted_vs = jnp.asarray(pred_vs[i])
            predicted_dqs = jnp.asarray(pred_dqs[i])
            loss += mse_loss_fn(expected_vs[i], predicted_vs) \
                  + mse_loss_fn(expected_dqs[i], predicted_dqs)

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
    for g in graphs:
        pred_graph = curr_state.apply_fn(state.params, g)
        pred_vs = jnp.array(pred_graph.nodes).reshape(-1)
        pred_dqs = jnp.array(pred_graph.edges).reshape(-1)

    _, expected_dqs, _, expected_vs = get_labelled_data(graphs)
    
    loss = 0
    for i in range(len(graphs)):
        predicted_vs = jnp.asarray(pred_vs[i])
        predicted_dqs = jnp.asarray(pred_dqs[i])
        loss += mse_loss_fn(expected_vs[i], predicted_vs) \
              + mse_loss_fn(expected_dqs[i], predicted_dqs)
    
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
    parser.add_argument('--data', type=str, default=default_dataset_path)
    args = parser.parse_args()

    train(args)