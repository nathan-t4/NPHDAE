import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from argparse import ArgumentParser
from scripts.graph_builder import DMSDGraphBuilder

def plot_trajs(args):
    gb = DMSDGraphBuilder(args.path, True, True, 'acceleration', 5)
    eval_gb = DMSDGraphBuilder(args.eval_path, True, True, 'acceleration', 5)
    print('eval masses', eval_gb._m)
    traj = gb._qs[0]
    ts = jnp.linspace(0, 15, len(traj))

    plt.title('Trajectories')
    plt.xlabel('t [s]')
    plt.ylabel('q [m]')

    red_patch = mpatches.Patch(color='red', label='train m0')
    yellow_path = mpatches.Patch(color='yellow', label='train m1')
    blue_patch = mpatches.Patch(color='blue', label='eval m0')
    green_patch = mpatches.Patch(color='green', label='eval m1')


    for i in range(20):
        plt.plot(ts, gb._qs[i,:,0], '--r', alpha=0.1)
        plt.plot(ts, gb._qs[i,:,1], '--y', alpha=0.1)
        plt.plot(ts, eval_gb._qs[i,:,0], '--b', alpha=0.1)
        plt.plot(ts, eval_gb._qs[i,:,1], '--g', alpha=0.1)
    # plt.legend()
    plt.legend(handles=[red_patch, yellow_path, blue_patch, green_patch])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--eval_path', type=str)
    args = parser.parse_args()

    gb = DMSDGraphBuilder(args.path, True, True, 'acceleration', 5, 1)
    graphs = gb.get_graph_batch(jnp.array([0,0]), jnp.array([6,7]))
    print(gb._data.shape)
    print(graphs)

    # plot_trajs(args)
    