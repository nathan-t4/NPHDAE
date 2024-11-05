import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from environments.random_env import PHDAEEnvironment
from environments.dgu_triangle_dae import DGU_TRIANGLE_PH_DAE

"""
    A simple circuit that exhibits chaotic behavior. 

    https://en.wikipedia.org/wiki/Chua%27s_circuit

    The current across the nonlinear resistor R2 (the Chua diode) is:
    I2(V2) = m1 V2 + 0.5 (m0 - m1) (|V2 + 1| - |V2 - 1|)
"""

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', '-d', required=True, type=str)
    args = parser.parse_args()

    dataset_type = args.data

    assert(dataset_type in ['train', 'val', 'test'])

    AC = jnp.array([[0.0, 1.0],
                    [1.0, 0.0]])
    AR = jnp.array([[-1.0, 0.0],
                    [1.0, 1.0]])
    AL = jnp.array([[-1.0], [0.0]])
    AV = jnp.array([[0.0], [0.0]])
    AI = jnp.array([[0.0], [0.0]])

    R1 = 1.0
    m0 = -8/7
    m1 = -5/7
    alpha = 15.6
    beta = 28
    C = 1 / alpha
    L = 1 / beta

    def grad_H_func(phi, jax_key, params=None):
        return phi / L

    def r_func(delta_V, jax_key, params=None):
        I2 = lambda dV : m1 * dV + 0.5 * (m0 - m1) * (jnp.abs(dV + 1) - jnp.abs(dV - 1))
        return jnp.array([delta_V[0] / R1, I2(delta_V[1])])
    
    def q_func(delta_V, jax_key, params=None):
        # Capacitance of C2 must be 1
        return jnp.array([C * delta_V[0], delta_V[1]])
    
    def u_func(t, jax_key, params=None):
        return jnp.array([])
    
    dt = 1e-3

    if dataset_type == 'train':
        seed = 41
        trajectory_num_steps = 1000
        num_trajectories = 500
    elif dataset_type == 'val':
        seed = 42
        trajectory_num_steps = 800
        num_trajectories = 20
    else:
        seed = 0
        trajectory_num_steps = 100
        num_trajectories = 2
    
    env = PHDAEEnvironment(AC, AR, AL, AV, AI, grad_H_func, q_func, r_func, u_func, dt=dt, seed=seed, name='Chua')
    
    curdir = os.path.abspath(os.path.curdir)
    save_dir = os.path.abspath(os.path.join(curdir, 'chua_data'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset = env.gen_dataset(
        z0_init_lb=jnp.array([-0.5, -0.5, -0.5, 0.0, 0.0]),
        z0_init_ub=jnp.array([0.5, 0.5, 0.5, 0.0, 0.0]),
        trajectory_num_steps=trajectory_num_steps,
        num_trajectories=num_trajectories,
        save_str=save_dir,
        save_name=dataset_type,
    )

    traj = dataset['state_trajectories'][0,:,:]
    T = jnp.arange(dataset['state_trajectories'].shape[1]) * env.dt
    fig, ax = plt.subplots(3, 1, figsize=(15,25))
    print(traj.shape)
    ax[0].plot(T, traj[:,0:2], label=['q1', 'q2'])
    ax[1].plot(T, traj[:,2], label='phi')
    ax[2].plot(T, traj[:,3:5], label=['e1', 'e2'])

    for x in ax: x.legend()

    plt.show()