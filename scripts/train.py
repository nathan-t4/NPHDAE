import jax
import optax
import functools
from clu import metric_writers
import numpy as np
import flax.linen as nn
from flax.training import train_state

from typing import Tuple
from argparse import ArgumentParser
import matplotlib.pyplot as plt

# from ..models.gnn.build_graph import *

from scripts.build_graph import *
from scripts.models import *

    
def compute_loss(params, graph: jraph.GraphsTuple,
                 senders: jnp.ndarray, receivers: jnp.ndarray,
                 labels: jnp.ndarray, net) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # loss function will be predicted position / momentum / Lagrangian / Hamiltonian vs actual?
    # predicted vs expected acceleration
    loss = 0
    accuracy = 0
    return loss, accuracy # TODO: return energy (e.g. Hamiltonian)

def train(args):
    data = np.load(args.data, allow_pickle=True)
    graph = build_double_spring_mass_graph(data)
    draw_jraph_graph_structure(graph)
    # plt.show()
    
    # TODO: use ml_collections?
    # config = {}
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    init_graph = graph
    init_net = GraphNet()
    params = jax.jit(init_net.init)(init_rng, init_graph)

    # Create writer for logs
    writer = metric_writers.create_default_writer(logdir=f'results')

    # Create the optimizer
    tx = optax.adam(learning_rate=1e-4)

    # Create the training state
    net = GraphNet()
    state = train_state.TrainState.create(
        apply_fn=net.apply, params=params, tx=tx,
    )

    # Create the evaluation net
    eval_net = GraphNet()
    eval_state = state.replace(apply_fn=eval_net.apply)

    init_epoch = 0
    num_epochs = 200
    # # TODO: load data, split to train and test set, 
    for epoch in range(init_epoch, num_epochs):
        print(f"Training step {epoch}")
        with jax.profiler.StepTraceAnnotation('train', step_num=epoch):
            train_step(state, graph)
      
    # # testing
    # test_loss, test_preds = compute_loss()

@jax.jit
def train_step(state: train_state.TrainState, graph: jraph.GraphsTuple):
    # loss_fn = compute_loss(graph)
    # grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    # _, grads = grad_fn(state.params, graph)

    # state = state.apply_gradients(grads=grads)

    return state

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    args = parser.parse_args()

    train(args)