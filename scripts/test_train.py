import jax
import os
from scripts.train import *
from scripts.models import GraphNet
from utils.data_utils import load_data_jnp


def test_update_fns(path: str | os.PathLike): 
    """ Verify graph update functions work as intended """   
    init_graphs = build_graph(data=path, batch_size=1, render=False)
    g = init_graphs[0]

    net = GraphNet()
    rng = jax.random.PRNGKey(0)
    params = net.init(rng, g)

    for i in range(10):
        prev_g = g
        g = net.apply(params, prev_g)
        print(f"NODES {i}:", g.nodes)
        print(f"EDGES {i}:", g.edges)
        print(f"GLOBALS {i}:", g.globals)

        assert g.globals[0] - prev_g.globals[0] == 1, "Time is not being incremented!"

def test_train_step(path: str | os.PathLike):
    init_graphs = build_graph(data=path, batch_size=1, render=False)

    lr = 1e-3

    net = GraphNet()
    rng = jax.random.PRNGKey(0)
    params = net.init(rng, init_graphs)

    tx = optax.adam(learning_rate=lr)
    state = train_state.TrainState.create(
        apply_fn=net.apply, params=params, tx=tx,
    )

    for i in range(200):
        state, metrics = train_step(state=state, graphs=init_graphs, verbose=True)


if __name__ == '__main__':
    default_dataset_path = 'results/double_mass_spring_data/Double_Spring_Mass_2023-12-26-14-29-05.pkl'
    
    # test_update_fns(path=default_dataset_path)

    test_train_step(path=default_dataset_path)