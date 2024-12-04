import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from environments.random_env import PHDAEEnvironment
from environments.dgu_triangle_dae import DGU_TRIANGLE_PH_DAE

"""
    Simulation of the FitzHugh-Nagumo circuit, a simple circuit with a nonlinear resistor.

    http://www.scholarpedia.org/article/FitzHugh-Nagumo_model

    The current across the nonlinear resistor R1 is I1(V) = V**3 / 3 - V
"""
jax.config.update('jax_platform_name', 'cpu')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', '-d', required=True, type=str)
    args = parser.parse_args()

    dataset_type = args.data

    assert(dataset_type in ['train', 'train_less_data', 'val', 'test'])

    AC = jnp.array([[1.0], [0.0], [0.0]])
    AR = jnp.array([[1.0, -1.0],
                    [0.0, 1.0],
                    [0.0, 0.0]])
    AL = jnp.array([[0.0], [0.0], [-1.0]])
    AV = jnp.array([[0.0], [-1.0], [1.0]])
    AI = jnp.array([[1.0], [0.0], [0.0]])

    R1 = 1.0
    R2 = 0.8
    C = 1.0
    L = 1 / 0.08
    E = -0.7
    J_mag = 1.0


    def grad_H_func(phi, jax_key, params=None):
        return phi / L

    def r_func(delta_V, jax_key, params=None):
        I1 = lambda V : V**3 / 3 - V
        return jnp.array([I1(delta_V[0]), delta_V[1] / R2])
    
    def q_func(delta_V, jax_key, params=None):
        return C * delta_V
    
    def u_func(t, jax_key, params=None):
        J = J_mag # * jnp.sin(1*t)
        return jnp.array([J, E])
    
    dt = 1e-1

    if dataset_type == 'train':
        seed = 3
        trajectory_num_steps = 1000
        num_trajectories = 1000
    elif dataset_type =='train_less_data':
        seed = 3
        trajectory_num_steps = 1000
        num_trajectories = 30
    elif dataset_type == 'val':
        seed = 4
        trajectory_num_steps = 10000
        num_trajectories = 10
    else:
        seed = 4
        trajectory_num_steps = 1000
        num_trajectories = 2
    
    env = PHDAEEnvironment(AC, AR, AL, AV, AI, grad_H_func, q_func, r_func, u_func, dt=dt, seed=seed, name='fitz_hugh_nagano')
    
    curdir = os.path.abspath(os.path.curdir)
    save_dir = os.path.abspath(os.path.join(curdir, 'fitz_hugh_nagano_data'))

    save_name = dataset_type + ".pkl"
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset = env.gen_dataset(
        z0_init_lb=jnp.array([-3.0 / C, -3.0 / L, 0.0, 0.0, 0.0, 0.0]),
        z0_init_ub=jnp.array([3.0 / C, 3.0 / L, 0.0, 0.0, 0.0, 0.0]),
        trajectory_num_steps=trajectory_num_steps,
        num_trajectories=num_trajectories,
        save_str=save_dir,
        save_name=save_name,
    )

    traj = dataset['state_trajectories'][0,:,:]
    T = jnp.arange(dataset['state_trajectories'].shape[1]) * env.dt
    fig, ax = plt.subplots(3, 1, figsize=(15,25))
    print(traj.shape)
    ax[0].plot(T, traj[:,0:2], label=['q1', 'phi1'])
    ax[1].plot(T, traj[:,2:5], label=['e1', 'e2', 'e3'])
    ax[2].plot(T, traj[:,5], label='jv1')

    for x in ax: x.legend()

    plt.show()