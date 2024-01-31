import os
import jax

import tensorflow as tf
import numpy as np
import jax.numpy as jnp

from argparse import ArgumentParser
from utils.custom_types import GraphLabels

def load_data_jnp(data: str | dict | os.PathLike) -> jnp.ndarray:
    """
        Load experimental data to tensorflow dataset
    """
    if type(data) is str:
        data = np.load(data, allow_pickle=True)
    
    state = data['state_trajectories']
   
    qs = jnp.stack((jnp.zeros(shape=jnp.shape(state[:,:,0])),
                        state[:,:,0], 
                        state[:,:,2]), 
                        axis=-1) # q_wall, q1, q2
    
    # relative positions 
    dqs = jnp.stack((qs[:,:,1] - qs[:,:,0], 
                     qs[:,:,2] - qs[:,:,1]), 
                     axis=-1)
    
    ps = jnp.stack((jnp.zeros(shape=jnp.shape(state[:,:,0])), 
                    state[:,:,1], 
                    state[:,:,3]),
                    axis=-1) # p_wall, p1, p2
    
    # The dataset has dimensions [num_trajectories, num_timesteps, (qs, dqs, ps)]
    data = jnp.concatenate((qs, dqs, ps), axis=-1)
    # Stop gradient for data
    data = jax.lax.stop_gradient(data)

    return data

def load_data_tf(data: str | dict) -> tf.data.Dataset:
    """
        Load experimental data to tensorflow dataset
    """    
    data = load_data_jnp(data=data)
    data = tf.data.Dataset.from_tensor_slices(data)

    return data

""" Test load_data() and batching with tf.data.Dataset  """
if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    data = np.load(args.path, allow_pickle=True)

    data_tf = load_data_tf(data)

    # print(f'Dataset specs: {data_tf.element_spec}')
    
    test_batch = data_tf.batch(batch_size=3, deterministic=False, drop_remainder=True)
    test_batch_np = list(test_batch.as_numpy_iterator())
    print(f'Test batch length: {len(test_batch_np)}')
    print(f'First batch shape: {np.shape(test_batch_np[0])}')