import jax
import optax
import os

import numpy as np
import flax.linen as nn
import matplotlib.pyplot as plt

from time import strftime
from argparse import ArgumentParser
from typing import Tuple

from absl import logging
from clu import metric_writers
from clu import metrics
from clu import checkpoint
from clu import periodic_actions
from flax.training import train_state

from scripts.build_graph import *
from scripts.data_utils import load_data
from scripts.models import *
from utils.custom_types import GraphLabels
from utils.logging_utils import *


def build_graph(data: str, render: bool, batch_size: int = 1):
    """
        Returns graph generated using the dataset config

        :param path: path to dataset
        :param render: whether to render graph using networkx
    """
    data = np.load(data, allow_pickle=True)
    num_trajectories = np.shape(data['state_trajectories'])[1]
    # TODO: start from zero, do next step prediction only (instead of random times)
    # TODO: curriculum learning?
    # rnd_times = np.random.randint(low=0, high=num_trajectories-1, size=batch_size)
    rnd_times = [0] * batch_size
    # TODO: efficiently batch
    graphs = []
    for i in rnd_times:
        graphs.append(build_double_spring_mass_graph(data, t=int(i)))

    if render:
        draw_jraph_graph_structure(graphs[0])
        plt.show()

    return graphs

def train(args):
    init_graphs = build_graph(data=args.data, batch_size=1, render=False)

    # training params
    lr = 1e-4
    eval_every_steps = 20
    log_every_steps = 10
    checkpoint_every_steps = 100
    num_train_steps = 2000

    work_dir = os.path.join(os.curdir, f'results/gnn/{strftime("%Y%m%d-%H%M%S")}')

    ds = load_data(data=args.data)
    # batch
    ds = ds.batch(batch_size=1, deterministic=False, drop_remainder=True)

    # TODO: split to training and validation sets
    # train_ds = train_ds.batch(batch_size=1, deterministic=False, drop_remainder=True)
    # test_ds = test_ds.batch(batch_size=1, deterministic=False, drop_remainder=True)
    # config = {}

    # Create key and initialize params
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    init_graph = init_graphs[0]
    init_net = GraphNet()
    params = jax.jit(init_net.init)(init_rng, init_graph)

    # Create writer for logs
    log_dir = os.path.join(work_dir, 'log')
    writer = metric_writers.create_default_writer(logdir=log_dir)

    # Create the optimizer
    tx = optax.adam(learning_rate=lr)

    # Create the training state
    net = GraphNet()
    state = train_state.TrainState.create(
        apply_fn=net.apply, params=params, tx=tx,
    )

    # Setup checkpointing for model
    checkpoint_dir = os.path.join(work_dir, 'checkpoints')
    ckpt = checkpoint.Checkpoint(checkpoint_dir)
    state = ckpt.restore_or_initialize(state)
    init_epoch = int(state.step) + 1

    # Create the evaluation net
    eval_net = GraphNet()
    eval_state = state.replace(apply_fn=eval_net.apply)

    # Create logger to report training progress
    report_progress = periodic_actions.ReportProgress(num_train_steps=num_train_steps, writer=writer)

    print("Starting training")
    train_metrics = None
    # Training loop
    for epoch in range(init_epoch, num_train_steps):
        with jax.profiler.StepTraceAnnotation('train', step_num=epoch):
            for data in ds: # batch? (w.r.t. to trajectories)
                data_np = np.squeeze(data.numpy())
                # what is important is to train at different timesteps and trajectories. the actual graph doesn't matter - we are training the message passing functions only (not anything related to the graph!)
                batch_graphs = build_graph(data=args.data, batch_size=1, render=False)
                state, metrics_update = train_step(state, batch_graphs[0], data_np)

                # Update metrics
                if train_metrics is None:
                    train_metrics = metrics_update
                else:
                    train_metrics = train_metrics.merge(metrics_update)
        
        print(f"Training step {epoch} - training loss {train_metrics.compute()}")

        report_progress(epoch)

        is_last_step = (epoch == num_train_steps - 1)

        # if epoch % eval_every_steps == 0 or is_last_step:
        #     eval_state = eval_state.replace(params=state.params)
        #     with report_progress.timed('eval'):
        #         eval_metrics = eval_model(eval_state)
        #         writer.write_scalars(epoch, add_prefix_to_keys(eval_metrics.compute(), 'eval'))

        if epoch % log_every_steps == 0 or is_last_step:
            writer.write_scalars(epoch, add_prefix_to_keys(train_metrics.compute(), 'train'))
            train_metrics = None

        if epoch & checkpoint_every_steps == 0 or is_last_step:
            with report_progress.timed('checkpoint'):
                ckpt.save(state)
            
    # TODO: validation loop

@jax.jit
def train_step(state: train_state.TrainState, 
               graph: jraph.GraphsTuple, 
               train_data: np.ndarray) -> Tuple[train_state.TrainState, metrics.Collection]:

    def loss_fn(params, graph: jraph.GraphsTuple, data: np.ndarray, rngs=None):
        curr_state = state.replace(params=params)
        pred_graph = curr_state.apply_fn(state.params, graph, rngs=rngs)
        pred_vs = pred_graph.nodes
        
        t = jnp.array(pred_graph.globals[0], int)
        masses = pred_graph.globals[1:4]
        # expected x, dx, and p at time t
        expected_qs = data[t,0:3] 
        expected_dqs = data[t,3:5]
        expected_ps = data[t,5:8]
        expected_vs = jnp.true_divide(expected_ps, masses) # component wise division
        
        # exclude training for wall node?
        # pred_qs = pred_vs[1,:]
        # expected_qs = expected_vs[1:]

        loss = jnp.linalg.norm(expected_vs - pred_vs)

        return loss, pred_graph.globals
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, pred_graph_globals), grads = grad_fn(state.params, graph, train_data)
    state = state.apply_gradients(grads=grads)

    metrics = TrainMetrics.single_from_model_output(loss=loss)

    return state, metrics

@jax.jit
def eval_step(state: train_state.TrainState,
              graph: jraph.GraphsTuple,
              eval_data: np.ndarray) -> metrics.Collection:
    pred_graph = state.apply_fn(state.params, graph, rngs=None)
    loss = 0.0
    return EvalMetrics.single_from_model_output(loss=loss)

def eval_model(state: train_state.TrainState):
    return {}

if __name__ == '__main__':
    default_dataset_path = 'results/double_mass_spring_data/Double_Spring_Mass_2023-12-26-14-29-05.pkl'

    parser = ArgumentParser()
    parser.add_argument('--data', type=str, default=default_dataset_path)
    args = parser.parse_args()

    train(args)