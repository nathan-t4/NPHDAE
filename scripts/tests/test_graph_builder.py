import jax.numpy as jnp
from argparse import ArgumentParser
from scripts.graph_builder import DMSDGraphBuilder

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    gb = DMSDGraphBuilder(args.path, True, True, 'acceleration', 5)
    graphs = gb.get_graph_batch(jnp.array([0,0]), jnp.array([6,7]))
    print(gb._data.shape)
    # print(graphs)