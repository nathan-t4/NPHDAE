import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import animation
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()
    
    data = np.load(args.dir, allow_pickle=True)
    params = data['config']
    trajs = data['state_trajectories']
    N = int(trajs.shape[2] // 2)
    control = np.array(data['control_inputs'])

    def mass_spring_damper_plotter(traj_idx=0):
        fig, ax = plt.subplots()
        ax.set_xlim(-2, np.max(trajs[0,:,::2]) + 2)
        ax.set_ylim(-0.5,0.5)
        ax.grid()
        traj = trajs[traj_idx]
        ms = [patches.Rectangle((traj[0,2*i], 0), 0.5, 0.1) for i in range(N)]

        def init():
            for i in range(N):
                ax.add_patch(ms[i])
            return *ms,

        def animate(i):
            for j in range(N):
                ms[j].set_xy([traj[i,2*j], 0])
            return *ms,

        anim = animation.FuncAnimation(fig, animate, init_func=init, interval=1, frames=len(traj), blit=True, repeat=True)
        plt.show()

    mass_spring_damper_plotter(10)
    # print(f'Mass of trajectory 0 to 5: {params["m"][0:5]}')

    print(f'Control input of trajectory zero: {control[0]}')
    print(f'Masses: {params["m"]}')
    print(f'Mass max {np.max(params["m"])} and min {np.min(params["m"])}')
    print(f'Spring max {np.max(params["k"])} and min {np.min(params["k"])}')
    print(f'Damper max {np.max(params["b"])} and min {np.min(params["b"])}')
    print(f'Control max {np.max(control)} and min {np.min(control)} and mean {np.mean(control)}')
    print(f'Position initial conditions max {np.max(trajs[:,0,::2])} and min {np.min(trajs[:,0,::2])}')
    print(f'Momenta initial conditions max {np.max(trajs[:,0,1::2])} and min {np.min(trajs[:,0,1::2])}')

    fig,(ax_1, ax_2) = plt.subplots(2, 1)
    T = np.arange(len(control[0]))
    for i in range(10):
        ax_1.plot(T, control[i,:,1::2], label=f'{i}') 
        # ax_1.plot(T, control[i], label=f'{i}')
    ax_1.legend()
    T = np.arange(len(trajs[0]))
    for i in range(10,11):
        label_fn = lambda suffix : [f'traj {i}: {suffix} {k}' for k in range(N)]
        ax_2.plot(T, trajs[i,:,::2], label=label_fn('pos'))
        ax_2.plot(T, trajs[i,:,1::2], label=label_fn('mom'))
    ax_2.legend()
    plt.show()