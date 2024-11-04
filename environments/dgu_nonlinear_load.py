import sys

sys.path.append('../')
from jax._src.random import PRNGKey as PRNGKey
import jax.numpy as jnp
import jax

import os
import time
import matplotlib.pyplot as plt
from environments.random_env import PHDAEEnvironment

if __name__ == '__main__':
    AC = jnp.array([[0.0], [0.0], [1.0]])
    AR = jnp.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0]])
    AL = jnp.array([[0.0], [1.0], [-1.0]])
    AV = jnp.array([[1.0], [0.0], [0.0]])
    AI = jnp.array([[0.0], [0.0], [-1.0]])

    dt = 1e-5
    R = 0.2; L = 1.8e-3; C = 2.2e-3
    I_magnitude = 1.0
    V_magnitude = 100.0
    init_charge_range = jnp.array([-1.0, 1.0]) * 1e-3
    init_flux_range = jnp.array([-1.0, 1.0]) * 1e-3

    def r_func(delta_V, jax_key, params=None):
        return jnp.array([delta_V[0] / R, (delta_V[1] * jnp.cos(delta_V[1])) / R])
    
    def q_func(delta_V, jax_key, params=None):
        return C * delta_V
    
    def grad_H_func(phi, jax_key, params=None):
        return phi / L
    
    def u_func(t, jax_key, params):
        jax_key, ik, vk = jax.random.split(jax_key, 3)
        # i = jax.random.uniform(ik, minval=0.0, maxval=1.0)
        # v = jax.random.uniform(vk, minval=0.0, maxval=1.0)
        # return jnp.array([I_magnitude * i, V_magnitude * v])
        # return jnp.array([I_magnitude, V_magnitude])
        return jnp.array([I_magnitude, V_magnitude * jnp.sin(1e3 * t)])
    
    env = PHDAEEnvironment(AC, AR, AL, AV, AI, grad_H_func, q_func, r_func, u_func, dt=dt)

    curdir = os.path.abspath(os.path.curdir)
    # save_dir = os.path.abspath(os.path.join(curdir, 'results/DGU_data'))
    save_dir = os.path.abspath(os.path.join(curdir, 'nonlinear_dgu_dae_data'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    t = time.time()
    print('starting simulation')

    # seed = 42 # for testing
    seed = 49 # for training
    dataset = env.gen_dataset(
        z0_init_lb=jnp.array(
            [init_charge_range[0], init_flux_range[0], V_magnitude, V_magnitude, 0.0, 0.0]
        ),
        z0_init_ub=jnp.array(
            [init_charge_range[1], init_flux_range[1], V_magnitude, V_magnitude, V_magnitude, V_magnitude]
        ),
        trajectory_num_steps=5000, # 1000 for training, 800 for testing.
        num_trajectories=1, # 500 for training, 20 for testing
        save_str=save_dir,
    )

    traj = dataset['state_trajectories'][0]
    plt.plot(traj, label=['q', 'phi', 'e1', 'e2', 'e3', 'jv'])
    plt.legend()
    plt.show()

    print(dataset['control_inputs'].shape)
    control = dataset['control_inputs'][0]
    plt.plot(control, label=['i', 'v'])
    plt.legend()
    plt.show()

    print(time.time() - t)