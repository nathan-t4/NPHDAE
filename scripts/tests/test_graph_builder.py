import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from argparse import ArgumentParser
from scripts.graph_builder import DMSDGraphBuilder

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--eval_path', type=str, required=True)
    args = parser.parse_args()

    gb = DMSDGraphBuilder(args.path, True, True, 'acceleration', 5)
    # graphs = gb.get_graph_batch(jnp.array([0,0]), jnp.array([6,7]))
    # print(gb._data.shape)
    # print(graphs)
    eval_gb = DMSDGraphBuilder(args.eval_path, True, True, 'acceleration', 5)

    traj = gb._qs[0]
    ts = jnp.linspace(0, 15, len(traj))

    plt.title('Trajectories')
    plt.xlabel('t [s]')
    plt.xlabel('q [m]')

    red_patch = mpatches.Patch(color='red', label='train')
    blue_patch = mpatches.Patch(color='blue', label='eval')

    for i in range(20):
        plt.plot(ts, gb._qs[i,:,0], '--r', alpha=0.1, label='0' * i)
        plt.plot(ts, gb._qs[i,:,1], '--r', alpha=0.1, label='1' * i)
        plt.plot(ts, eval_gb._qs[i,:,0], '--b', alpha=0.1, label='e' + '0' * i)
        plt.plot(ts, eval_gb._qs[i,:,1], '--b', alpha=0.1, label='e' + '1' * i)
    # plt.legend()
    plt.legend(handles=[red_patch, blue_patch])
    plt.tight_layout()
    plt.show()