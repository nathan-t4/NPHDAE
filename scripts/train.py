import jax
import optax
import os

import numpy as np
import flax.linen as nn

from time import strftime
from clu import metric_writers
from flax.training import train_state

from utils.custom_types import GraphLabels
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from scripts.build_graph import *
from scripts.data_utils import load_data
from scripts.models import *

def build_graph(data: str, render: bool):
    """
        Returns graph generated using the dataset config

        :param path: path to dataset
        :param render: whether to render graph using networkx
    """
    data = np.load(data, allow_pickle=True)
    graph = build_double_spring_mass_graph(data)
    if render:
        draw_jraph_graph_structure(graph)
        plt.show()

    return graph

def train(args):
    # TODO: batch graphs with different time steps!
    graph = build_graph(data=args.data, render=True)

    # training params
    lr = 1e-4
    eval_every_steps = 20
    init_epoch = 0
    num_epochs = 200

    ds = load_data(data=args.data)
    # train_ds, test_ds = train_test_split(ds, test_size=0.2, shuffle=False)
    # batch
    ds = ds.batch(batch_size=1, deterministic=False, drop_remainder=True)

    # train_ds = train_ds.batch(batch_size=1, deterministic=False, drop_remainder=True)
    # test_ds = test_ds.batch(batch_size=1, deterministic=False, drop_remainder=True)
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
            for data in ds: # batch? (w.r.t. to trajectories)
                data_np = np.squeeze(data.numpy())
                print('shape: ', np.shape(data_np))
                state = train_step(state, graph, data_np)

        if epoch % eval_every_steps == 0:
            eval_metrics = eval_model(eval_state)
            
    # TODO: validation loop


def train_step(state: train_state.TrainState, 
               graph: jraph.GraphsTuple, 
               batch: GraphLabels) -> train_state.TrainState:

    def loss_fn(params, graph: jraph.GraphsTuple, data: np.ndarray, rngs=None):
        # loss function =  predicted position/momentum/acceleration/Lagrangian/Hamiltonian - actual?
        losses = []
        for i in range(jnp.shape(data)[0]): # loop for all time
            curr_state = state.replace(params=params)
            pred_graph = curr_state.apply_fn(state.params, graph, rngs=rngs)
            pred_qs = pred_graph.nodes[:,1:]
            # TODO: t is not being updated...
            t = pred_graph.globals[0]

            # expected x, dx, and p at time t
            expected_qs = data[t,0:3] 
            expected_dqs = data[t,3:5]
            expected_ps = data[t,5:8]

            loss = jnp.linalg.norm(expected_qs - pred_qs) # + expected_ps - pred_ps
            losses.append(loss) 
            print(f"Time {t}: expected {expected_qs} vs prediction {pred_qs}")
            print(f"Loss: {loss}")

        loss = jnp.mean(jnp.array(losses))
        accuracy = jnp.array(0.)

        # graph.globals[0] = t + 1

        # TODO: make sure updating graph of correct traj_idx. batch?

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