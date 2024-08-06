from ast import mod
import os, sys
sys.path.append('../')

from helpers.model_factory import get_model_factory

import jax
import jax.numpy as jnp
import json, pickle


def load_config_file(experiment_save_path):
    config_file_str = os.path.abspath(os.path.join(experiment_save_path, 'config.json'))
    with open(config_file_str, 'r') as f:
        config = json.load(f)
    return config

def load_dataset(experiment_save_path):

    config = load_config_file(experiment_save_path)
    dataset_config = config['dataset_setup']

    from helpers.dataloader import load_dataset_from_setup
    train_dataset, test_dataset = load_dataset_from_setup(dataset_config)

    datasets = {'train_dataset' : train_dataset, 'test_dataset' : test_dataset}

    return datasets

def load_model(experiment_save_path):

    # Load the model config and re-construct the model
    config = load_config_file(experiment_save_path)
    model_setup = config['model_setup']

    model = get_model_factory(model_setup).create_model(jax.random.PRNGKey(0))

    # Load the "Run" json file to get the artifacts path
    run_file_str = os.path.abspath(os.path.join(experiment_save_path, 'run.json'))
    with open(run_file_str, 'r') as f:
        run = json.load(f)

    artifacts_path = os.path.abspath(os.path.join(experiment_save_path, 'model_params.pkl'))
    with open(artifacts_path, 'rb') as f:
        params = pickle.load(f)
    
    return model, params

def load_metrics(experiment_save_path):
    metrics_file_str = os.path.abspath(os.path.join(experiment_save_path, 'metrics.json'))
    with open(metrics_file_str, 'r') as f:
        metrics = json.load(f)
    return metrics

def compute_traj_err(true_traj, pred_traj):
    # per step err is 2 norm
    errs = (true_traj - pred_traj)**2
    errs = jnp.sum(errs, axis=1)
    errs = jnp.sqrt(errs)
    return errs

def predict_trajectory(model, params, initial_state, num_steps, dt, t_init=0.0):
    """
    Predict the trajectory of the model given the initial state.

    Parameters
    ----------
    model : 
        The model object that contains the forward function.
    params : 
        The parameters of the model.
    initial_state : 
        The initial state of the model.
    num_steps : 
        The number of steps to predict the trajectory for.
    """
    forward = model.forward
    predicted_traj = [initial_state]
    timesteps = [t_init]
    for i in range(num_steps - 1):
        curr_t = timesteps[-1]
        z = jnp.concatenate((predicted_traj[-1], jnp.array([curr_t]))).reshape(1, len(initial_state) + 1)
        predicted_traj.append(forward(params, z).reshape(len(initial_state),))
        timesteps.append(timesteps[-1] + dt)
    return jnp.array(predicted_traj), jnp.array(timesteps)

def compute_g_vals_along_traj(g, params, traj, timesteps):
    g_vals = []
    for t_ind in range(traj.shape[0]):
        t = timesteps[t_ind]
        z = traj[t_ind, :]
        x = z[0:2]
        y = z[2::]
        g_vals.append(g(x,y,t,params))

    g_vals = jnp.array(g_vals)
    g_vals_norm = jnp.sqrt(jnp.sum(g_vals**2, axis=1))

    return g_vals_norm, g_vals