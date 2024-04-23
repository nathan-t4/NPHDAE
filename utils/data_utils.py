import os
import jax

import tensorflow as tf
import numpy as np
import jax.numpy as jnp
import ml_collections

from argparse import ArgumentParser
from typing import Tuple

def load_spring_mass_damper_data(path: os.PathLike) -> Tuple[jnp.ndarray, ml_collections.ConfigDict]:
    """
        Load experimental data to tensorflow dataset
    """
    data = np.load(path, allow_pickle=True)
    
    state = data['state_trajectories']
    config = data['config']
    dt = config['dt']

    m = jnp.array([config['m1'], config['m2']])
   
    qs = jnp.stack((state[:,:,0], 
                    state[:,:,2]), 
                    axis=-1) # q_wall, q1, q2
    # relative positions 
    dqs = jnp.expand_dims(qs[:,:,1] - qs[:,:,0], axis=-1)
    
    ps = jnp.stack((state[:,:,1], 
                    state[:,:,3]),
                    axis=-1) # p_wall, p1, p2
    
    vs = ps / m
    accs = jnp.diff(vs, axis=1) / dt
    final_acc = jnp.expand_dims(accs[:,-1,:], axis=1) # duplicate final acceleration
    accs = jnp.concatenate((accs, final_acc), axis=1) # add copy of final acceleration to end of accs
    # The dataset has dimensions [num_trajectories, num_timesteps, (qs, dqs, ps, accs)]
    data = jnp.concatenate((qs, dqs, ps, accs), axis=-1)
    # Stop gradient for data
    data = jax.lax.stop_gradient(data)

    normalization_stats = ml_collections.ConfigDict()
    normalization_stats.position = ml_collections.ConfigDict({
        'mean': jnp.mean(qs),
        'std': jnp.std(qs),
    })
    normalization_stats.velocity = ml_collections.ConfigDict({
        'mean': jnp.mean(vs),
        'std': jnp.std(vs),
    })
    normalization_stats.acceleration = ml_collections.ConfigDict({
        'mean': jnp.mean(accs),
        'std': jnp.std(accs),
    })

    return data, normalization_stats

def load_data_tf(data: str) -> tf.data.Dataset:
    """
        Load experimental data to tensorflow dataset
    """    
    data = load_spring_mass_damper_data(data=data)
    data = tf.data.Dataset.from_tensor_slices(data)

    return data

""" Test load_data() and batching with tf.data.Dataset  """
if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    data, norm_stats = load_spring_mass_damper_data(args.path)
    print(f'First trajectory first acceleration: {data[0,0,-1]}')