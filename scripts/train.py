import jax
import optax
import functools
import os

import numpy as np
import flax.linen as nn

from time import strftime
from clu import metric_writers
from flax.training import train_state

import ml_collections
from ml_collections import config_dict

from typing import Tuple
from utils.custom_types import GraphLabels
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from scripts.build_graph import *
from scripts.models import *

def build_graph(path: str, render: bool):
    """
        Returns graph generated using the dataset config

        :param path: path to dataset
        :param render: whether to render graph using networkx
    """
    data = np.load(path, allow_pickle=True)
    graph = build_double_spring_mass_graph(data)
    if render:
        draw_jraph_graph_structure(graph)
        plt.show()

    return graph

def train(args):
    graph = build_graph(path=args.data, render=True)

    # training params
    lr = 1e-4
    eval_every_steps = 20
    init_epoch = 0
    num_epochs = 200

    training_data = None

    # config = {}
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    init_graph = graph
    init_net = GraphNet()
    params = jax.jit(init_net.init)(init_rng, init_graph)

    # Create writer for logs
    logdir = os.path.join(os.curdir, f'results/gnn/{strftime("%Y%m%d-%H%M%S")}')
    writer = metric_writers.create_default_writer(logdir=logdir)

    # Create the optimizer
    tx = optax.adam(learning_rate=lr)

    # Create the training state
    net = GraphNet()
    state = train_state.TrainState.create(
        apply_fn=net.apply, params=params, tx=tx,
    )

    # Create the evaluation net
    eval_net = GraphNet()
    eval_state = state.replace(apply_fn=eval_net.apply)

    # Training loop
    for epoch in range(init_epoch, num_epochs):
        # print(f"Training step {epoch}")

        # TODO: batch wrt to trajectory
        with jax.profiler.StepTraceAnnotation('train', step_num=epoch):
            for data in training_data: # batch
                state = train_step(state, graph, data)

        if epoch % eval_every_steps == 0:
            eval_metrics = eval_model(eval_state)
            
    # TODO: validation loop

@jax.jit
def train_step(state: train_state.TrainState, graph: jraph.GraphsTuple, batch: GraphLabels):

    def loss_fn(params, graph: jraph.GraphsTuple, labels: GraphLabels, rngs=None):
        # loss function will be predicted position / momentum / Lagrangian / Hamiltonian vs actual?
        # predicted vs expected acceleration
        curr_state = state.replace(params=params)
        
        pred_graph = state.apply_fn(state.params, graph, rngs=rngs)
        pred_qs = pred_graph.nodes[:,1:]
        # TODO: replace graph node features with actual positions, momentum, etc. at this timestep
        t = pred_graph.globals[0]
        expected_qs = labels['q']

        loss = jnp.mean(expected_qs - pred_qs) # + expected_ps - pred_ps
        # loss = jnp.array(0.)
        accuracy = jnp.array(0.)

        return loss, accuracy
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, _), grads = grad_fn(state.params, graph, batch)
    state = state.apply_gradients(grads=grads)

    return state

@jax.jit
def eval_model(state: train_state.TrainState):
    return 0.

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    args = parser.parse_args()

    train(args)