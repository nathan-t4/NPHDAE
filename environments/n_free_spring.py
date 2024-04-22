import os
import pickle
import jax
import jax.numpy as jnp
import numpy as np
from copy import deepcopy
from functools import partial
from tqdm import tqdm
from helpers.integrator_factory import integrator_factory
from environments.utils import *

def generate_trajectory(args, seed=0):
    rng = np.random.default_rng(seed)

    save_dir = os.path.join(os.curdir, 'results/free_spring/')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    default_config = {
        'dt': 0.01,
        'm': [1.0, 1.0],
        'k': [1.0],
        'b': [0],
        'y': [1.0],
    }

    N = len(default_config['m']) # num masses

    x0_lb = jnp.array([0, 0, default_config['y'][0], 0]) 
    x0_ub = 10 * jnp.array([0.1, 0, default_config['y'][0], 0])

    assert x0_lb.shape == x0_ub.shape

    def get_dynamics(m, k, y):
        f = jnp.array([[0, 1, 0, 0],
                    [-k[0] / m[0], 0, k[0] / m[0], 0],
                    [0, 0, 0, 1],
                    [k[0] / m[1], 0, -k[0] / m[1], 0]])
        b = jnp.array([0, -k[0] * y[0], 0, k[0] * y[0]])
        return f, b

    def control_policy(x, t, aux_data):
        if args.control == 'passive':
            return jnp.zeros((2*N))
        elif args.control == 'all_random_continuous':
            coefficients, scales, offsets = aux_data
            f = lambda k : jnp.sum(coefficients * jnp.cos(scales * k + offsets), axis=1)
            control = [0] * (2 * N)
            control[1::2] = f(t).squeeze()
            return jnp.array(control)
        else:
            raise NotImplementedError(f'Invalid control {args.control}')

    integrator = integrator_factory('rk4')
    timesteps = jnp.arange(args.trajectory_num_steps) * default_config['dt']

    def generate_random_trajectory(jax_key, num_trajectories, config):
        trajectory = []
        control_inputs = []
    
        masses = []
        ks = []
        bs = []
        ys = []
        for i in tqdm(range(num_trajectories), desc='Generating data'): 
            # Generate random key
            jax_key, rng_key = jax.random.split(jax_key)
            # Set random initial condition
            x0 = jax.random.uniform(rng_key, 
                                    shape=x0_lb.shape, 
                                    minval=x0_lb, 
                                    maxval=x0_ub)
            x_prev = x0
            states = [x_prev]
            controls = []

            # Randomize system parameters - in or out of distribution depending on if training or testing dataset
            if args.type == 'train':
                m = get_train_param(rng, config['m'], train_scales['m'])
                k = get_train_param(rng, config['k'], train_scales['k'])
                b = get_train_param(rng, config['b'], train_scales['b'])
                y = get_train_param(rng, config['y'], 0)
                coefficients = get_train_param(rng, jnp.zeros((N,1)), train_scales['control_coef'], (3,))
                scales       = get_train_param(rng, jnp.zeros((N,1)), train_scales['control_scale'], (3,))
                offsets      = get_train_param(rng, jnp.zeros((N,1)), train_scales['control_offset'], (3,))
            elif args.type == 'val':
                m = get_validation_param(rng, config['m'], train_scales['m'], val_scales['m'])
                k = get_validation_param(rng, config['k'], train_scales['k'], val_scales['k'])
                b = get_validation_param(rng, config['b'], train_scales['b'], val_scales['b'])
                y = get_validation_param(rng, config['y'], 0, 0)
                coefficients = get_validation_param(rng, jnp.zeros((N,1)), train_scales['control_coef'], val_scales['control_coef'], (3,))
                scales       = get_validation_param(rng, jnp.zeros((N,1)), train_scales['control_scale'], val_scales['control_scale'], (3,))
                offsets      = get_validation_param(rng, jnp.zeros((N,1)), train_scales['control_offset'], val_scales['control_offset'], (3,))

            masses.append(m)
            ks.append(k)
            bs.append(b)
            ys.append(y)

            aux_data = (jnp.array(coefficients), jnp.array(scales), jnp.array(offsets))

            f, b = get_dynamics(m, k, y)
            # dynamics to integrate
            dynamics = lambda x, t, f, b : jnp.matmul(f, x) + b + control_policy(x, t, aux_data)
            dynamics = jax.jit(dynamics)

            @jax.jit 
            def step(x, t):
                u_next = control_policy(x, t, aux_data)
                x_next = integrator(partial(dynamics, f=f, b=b), x, t, config['dt'])
                
                return x_next, (x_next, u_next)
            
            x_last, (states, controls) = jax.lax.scan(step, x_prev, timesteps)
            
            trajectory.append(states)
            control_inputs.append(controls)

        params = deepcopy(config)
        # Save parameters to config
        params['m'] = jnp.array(masses)
        params['k'] = jnp.array(ks)
        params['b'] = jnp.array(bs)
        params['y'] = jnp.array(ys)

        # Save to pkl file
        dataset = {}
        dataset['config'] = params
        dataset['state_trajectories'] = jnp.array(trajectory)
        dataset['control_inputs'] = jnp.array(control_inputs)
        dataset['timesteps'] = jnp.array(timesteps)

        return dataset

    train_scales = {
        'm': 0.1,
        'k': 0.1,
        'b': 0.0,
        'control_coef': 0.5,
        'control_scale': 0.5,
        'control_offset': 0.5,
    }
    val_scales = {
        'm': 0.1,
        'k': 0.1,
        'b': 0.0,
        'control_coef': 0.5,
        'control_scale': 0.5,
        'control_offset': 0.5,
    }

    jax_key = jax.random.key(seed)

    num_trajectories = args.n_train if args.type == 'train' else args.n_val
    dataset = {}
    bins = 10
    num_trajectories_per_bin = num_trajectories // bins

    for i in tqdm(range(bins), desc=''):
        key, jax_key = jax.random.split(jax_key)
        new_dataset = generate_random_trajectory(key, num_trajectories_per_bin, default_config)
        if i == 0:
            dataset = new_dataset
        else:
            dataset = merge_datasets(dataset, new_dataset, ('m', 'k', 'b'))

    # Save dataset
    save_path = os.path.join(save_dir, f"{args.type}_{num_trajectories}_{train_scales['m']}_{train_scales['control_coef']}.pkl")

    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--n_train', type=int, default=1500)
    parser.add_argument('--n_val', type=int, default=20)
    parser.add_argument('--trajectory_num_steps', type=int, default=1500)
    parser.add_argument('--control', type=str, required=True)
    args = parser.parse_args()

    generate_trajectory(args)