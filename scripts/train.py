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
from scripts.data_utils import *
from scripts.models import *
from utils.custom_types import GraphLabels
from utils.logging_utils import *
import utils.graph_utils as graph_utils


def build_graph(data: str, render: bool, batch_size: int = 1,
                add_undirected_edges: bool = True,
                add_self_loops: bool = True):
    """
        Returns graph generated using the dataset config

        :param path: path to dataset
        :param render: whether to render graph using networkx
    """
    data = np.load(data, allow_pickle=True)
    num_timesteps = np.shape(data['state_trajectories'])[1]
    # TODO: start from zero, do next step prediction only (instead of random times)
    # TODO: curriculum learning?
    rnd_times = np.random.randint(low=0, high=num_timesteps-1, size=batch_size)
    # TODO: efficiently batch
    graphs = []
    for i in rnd_times:
        graphs.append(build_double_spring_mass_graph(data, t=int(i), traj_idx=0))

    graphs = jraph.batch(graphs)

    if add_self_loops:
        graphs = graph_utils.add_self_loops(graphs)

    if add_undirected_edges:
        graphs = graph_utils.add_undirected_edges(graphs)

    if render:
        draw_jraph_graph_structure(graphs[0])
        plt.show()

    return graphs

def train(args):
    platform = jax.local_devices()[0].platform
    print('Running on platform:', platform.upper())

    init_graphs = build_graph(data=args.data, batch_size=1, render=False)

    # training params
    lr = 1e-3
    batch_size = 1
    eval_every_steps = 20
    log_every_steps = 10
    checkpoint_every_steps = 100
    num_train_steps = 2000

    work_dir = os.path.join(os.curdir, f'results/gnn/{strftime("%Y%m%d-%H%M%S")}')

    # Create key and initialize params
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    init_net = GraphNet()

    # Create writer for logs
    log_dir = os.path.join(work_dir, 'log')
    writer = metric_writers.create_default_writer(logdir=log_dir)

    # Create the optimizer
    tx = optax.adam(learning_rate=lr)

    # Create the training state
    net = GraphNet()
    # params = jax.jit(net.init)(init_rng, init_graphs)
    params = net.init(init_rng, init_graphs)
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
            # currently batching wrt to time only. 
            # Trajectory is fixed (see build_graph -- traj_idx = 0 for all graphs)
            graphs = build_graph(data=args.data, batch_size=batch_size, render=False)
            # graphs = jax.tree_util.tree_map(np.asarray, graphs)
            state, metrics_update = train_step(state, graphs)

            # Update metrics
            if train_metrics is None:
                train_metrics = metrics_update
            else:
                train_metrics = train_metrics.merge(metrics_update)
        
        print(f"Training step {epoch} - training loss {train_metrics.compute()}")

        report_progress(epoch)

        is_last_step = (epoch == num_train_steps - 1)

        if epoch % log_every_steps == 0 or is_last_step:
            writer.write_scalars(epoch, add_prefix_to_keys(train_metrics.compute(), 'train'))
            train_metrics = None

        if epoch % eval_every_steps == 0 or is_last_step:
            eval_state = eval_state.replace(params=state.params)
            with report_progress.timed('eval'):
                eval_metrics = eval_model(eval_state)
                writer.write_scalars(epoch, add_prefix_to_keys(eval_metrics.compute(), 'eval'))

        if epoch & checkpoint_every_steps == 0 or is_last_step:
            with report_progress.timed('checkpoint'):
                ckpt.save(state)
            
    # TODO: validation loop

def get_labelled_data(graphs: jraph.GraphsTuple, traj_idx: int = 0):
    # TODO: remove hardcoded path
    default_dataset_path = 'results/double_mass_spring_data/Double_Spring_Mass_2023-12-26-14-29-05.pkl'

    data = load_data_jnp(data=default_dataset_path)[traj_idx]

    t = graphs.globals[0]
    masses = (graphs.globals[1:4]).squeeze()
    # expected x, dx, and p at time t
    expected_qs = data[t,0:3] 
    expected_dqs = data[t,3:5]
    expected_ps = data[t,5:8]
    
    expected_vs = jnp.true_divide(expected_ps, masses) # component wise division
    expected_vs = expected_vs
        
    # exclude training for wall node?
    # pred_vs = pred_vs[1,:]
    # expected_vs = expected_vs[1:]

    assert jnp.shape(expected_vs) == jnp.shape(expected_ps), \
        "Error with component wise division using jax.numpy.true_divide"
    
    return expected_qs, expected_dqs, expected_ps, expected_vs        

@jax.jit
def mse_loss_fn(expected, predicted):
    return jnp.sum(optax.l2_loss(expected - predicted))

@jax.jit
def train_step(state: train_state.TrainState, 
               graphs: jraph.GraphsTuple,
               verbose: bool = False) -> Tuple[train_state.TrainState, metrics.Collection]:

    def loss_fn(params, graphs: jraph.GraphsTuple):
        curr_state = state.replace(params=params)
        pred_graphs = curr_state.apply_fn(curr_state.params, graphs)
        pred_vs = pred_graphs.nodes.reshape(-1)
        _, _, _, expected_vs = get_labelled_data(graphs)
        return mse_loss_fn(expected_vs, pred_vs)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False, allow_int=True)
    loss, grads = grad_fn(state.params, graphs)
    # Apply gradient to optimizer
    state = state.apply_gradients(grads=grads)
    # Update metrics
    metrics = TrainMetrics.single_from_model_output(loss=loss)
    
    return state, metrics

@jax.jit
def eval_step(state: train_state.TrainState,
              graphs: jraph.GraphsTuple) -> metrics.Collection:
    curr_state = state.replace(params=state.params)
    pred_graphs = curr_state.apply_fn(state.params, graphs)
    pred_vs = jnp.array(pred_graphs.nodes).reshape(-1)
    _, _, _, expected_vs = get_labelled_data(graphs)
    
    loss = mse_loss_fn(expected_vs, pred_vs)

    return EvalMetrics.single_from_model_output(loss=loss)

def eval_model(state: train_state.TrainState) -> metrics.Collection:
    eval_metrics = None

    graphs = build_graph(data=args.data, batch_size=1, render=False)
    metrics_update = eval_step(state, graphs)

    if eval_metrics is None:
        eval_metrics = metrics_update
    else:
        eval_metrics = eval_metrics.merge(metrics_update)

    return eval_metrics

if __name__ == '__main__':
    default_dataset_path = 'results/double_mass_spring_data/Double_Spring_Mass_2023-12-26-14-29-05.pkl'

    parser = ArgumentParser()
    parser.add_argument('--data', type=str, default=default_dataset_path)
    args = parser.parse_args()

    train(args)