import sys

sys.path.append('../')
from jax._src.random import PRNGKey as PRNGKey
import jax.numpy as jnp
import jax
import os
import time
import matplotlib.pyplot as plt
from environments.random_env import PHDAEEnvironment
from argparse import ArgumentParser

jax.config.update('jax_platform_name', 'cpu')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='default')
    args = parser.parse_args()

    dataset_type = args.data

    assert(dataset_type in ['test', 'default'])

    AC = jnp.array([[0.0], [0.0], [1.0]])
    AR = jnp.array([[-1.0], [1.0], [0.0]])
    AL = jnp.array([[0.0], [1.0], [-1.0]])
    AV = jnp.array([[1.0], [0.0], [0.0]])
    AI = jnp.array([[0.0], [0.0], [-1.0]])


    if dataset_type == 'default':
        dt = 1e-2
        R = 1.2; L = 1.8; C = 2.2
        I_magnitude = 0.1
        V_magnitude = 1.0
        init_charge_range = jnp.array([-1.0, 1.0])
        init_flux_range = jnp.array([-1.0, 1.0])

        train_seed = 2 # 41
        val_seed = 51 # 21
        train_num_steps = 500
        train_num_trajs = 30
        val_num_steps = 1000
        val_num_trajs = 10

        name = ''

    elif dataset_type == 'test':
        dt = 1e-2
        R = 1.2; L = 1.8; C = 2.2
        I_magnitude = 0.1
        V_magnitude = 100.0
        init_charge_range = jnp.array([-1.0, 1.0])
        init_flux_range = jnp.array([-1.0, 1.0])

        train_seed = 6 # 41 # 5
        val_seed = 7 # 21 # 100
        train_num_steps = 1000
        train_num_trajs = 1
        val_num_steps = 100
        val_num_trajs = 1

        name = 'test'


    def r_func(delta_V, jax_key, params=None):
        return delta_V / R
    
    def q_func(delta_V, jax_key, params=None):
        return C * delta_V
    
    def grad_H_func(phi, jax_key, params=None):
        return phi / L
    
    def u_func(t, jax_key, params):
        jax_key, ik, vk, omegak = jax.random.split(jax_key, 4)
        # i = jax.random.uniform(ik, minval=0.0, maxval=1.0)
        # omega = jax.random.uniform(omegak, minval=0.0, maxval=100.0)
        # v = jax.random.uniform(vk, minval=0.0, maxval=1.5)
        return jnp.array([I_magnitude, V_magnitude])
        # return jnp.array([I_magnitude, V_magnitude])
    
    curdir = os.path.abspath(os.path.curdir)
    save_dir = os.path.abspath(os.path.join(curdir, 'dgu_dae_data'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    t = time.time()
    print('starting simulation')

    save_name = name + ".pkl"

    z0_init_lb = jnp.array([init_charge_range[0], init_flux_range[0], V_magnitude, V_magnitude, 0.0, 0.0])
    z0_init_ub = jnp.array([init_charge_range[1], init_flux_range[1], V_magnitude, V_magnitude, V_magnitude, V_magnitude])

    train_seed = 0
    val_seed = 50
    while(True):
        try:
            train_seed += 1
            train_env = PHDAEEnvironment(AC, AR, AL, AV, AI, grad_H_func, q_func, r_func, u_func, dt=dt, seed=train_seed, name='distributed_generation_unit')

            train_dataset = train_env.gen_dataset(
                z0_init_lb=z0_init_lb,
                z0_init_ub=z0_init_ub,
                trajectory_num_steps=train_num_steps,
                num_trajectories=train_num_trajs,
                save_str=save_dir,
                save_name=f'train_{dataset_type}.pkl',
            )

            val_seed += 1
            val_env = PHDAEEnvironment(AC, AR, AL, AV, AI, grad_H_func, q_func, r_func, u_func, dt=dt, seed=val_seed, name='distributed_generation_unit')

            val_dataset = val_env.gen_dataset(
                z0_init_lb=z0_init_lb,
                z0_init_ub=z0_init_ub,
                trajectory_num_steps=val_num_steps,
                num_trajectories=val_num_trajs,
                save_str=save_dir,
                save_name=f'val_{dataset_type}.pkl',
            )
            break
        except:
            pass

    print(train_seed, val_seed)
            
    for i in range(train_num_trajs):
        traj = train_dataset['state_trajectories'][i]
        plt.plot(traj, label=['q', 'phi', 'e1', 'e2', 'e3', 'jv'])
    plt.savefig('example_train_traj.png')
    plt.clf()
    
    for i in range(train_num_trajs):
        control = train_dataset['control_inputs'][i]
        plt.plot(control, label=['i', 'v'])
    plt.savefig('example_train_control.png')
    plt.clf()

    for i in range(val_num_trajs):
        traj = val_dataset['state_trajectories'][i]
        plt.plot(traj, label=['q', 'phi', 'e1', 'e2', 'e3', 'jv'])
    plt.savefig('example_val_traj.png')
    plt.clf()

    for i in range(val_num_trajs):
        control = val_dataset['control_inputs'][i]
        plt.plot(control, label=['i', 'v'])
    plt.savefig('example_val_control.png')
    plt.clf()

    print(time.time() - t)