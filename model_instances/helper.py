import os
import json
import pickle
import jax
import jax.numpy as jnp
from models.ph_dae import PHDAE
from plotting.common import load_config_file
from helpers.model_factory import get_model_factory

def add_transmission_line(R=1, L=1):
    """
        Returns: tuple(PHDAE, PHDAE_params)
    """
    AC = jnp.array([[0.0], [0.0], [0.0]])
    AR = jnp.array([[1.0], [-1.0], [0.0]])
    AL = jnp.array([[0.0], [-1.0], [1.0]])
    AV = jnp.array([[0.0], [0.0], [0.0]])
    AI = jnp.array([[0.0], [0.0], [0.0]])

    def r_func(delta_V, params=None):
        return delta_V / R

    def q_func(delta_V, params=None):
        return None

    def grad_H_func(phi, params=None):
        return phi / L

    def u_func(t, params):
        return jnp.array([])

    transmission_line_phdae = PHDAE(AC, AR, AL, AV, AI, grad_H_func, q_func, r_func, u_func)
    transmission_line_params = None
    
    return transmission_line_phdae, transmission_line_params

def add_dgu(R=1, L=1, C=1, i_load=0.1, V=1.0):
    AC = jnp.array([[0.0], [0.0], [1.0]])
    AR = jnp.array([[-1.0], [1.0], [0.0]])
    AL = jnp.array([[0.0], [1.0], [-1.0]])
    AV = jnp.array([[1.0], [0.0], [0.0]])
    AI = jnp.array([[0.0], [0.0], [-1.0]])

    def r_func(delta_V, params=None):
        return delta_V / R

    def q_func(delta_V, params=None):
        return C * delta_V

    def grad_H_func(phi, params=None):
        return phi / L

    def u_func(t, params):
        return jnp.array([i_load, V])

    dgu_phdae = PHDAE(AC, AR, AL, AV, AI, grad_H_func, q_func, r_func, u_func)
    dgu_params = None

    return dgu_phdae, dgu_params

def add_ndgu(exp_file_name, i_load=0.1, v=1):
    sacred_save_path = os.path.abspath(os.path.join('../cyrus_experiments/runs/', exp_file_name, '1'))

    config = load_config_file(sacred_save_path)
    model_setup = config['model_setup']
    model_setup['u_func_freq'] = 0.0
    model_setup['u_func_current_source_magnitude'] = i_load
    model_setup['u_func_voltage_source_magnitude'] = v
    ndgu_model = get_model_factory(model_setup).create_model(jax.random.PRNGKey(0))

    # Load the "Run" json file to get the artifacts path
    run_file_str = os.path.abspath(os.path.join(sacred_save_path, 'run.json'))
    with open(run_file_str, 'r') as f:
        run = json.load(f)

    # Load the params for model 1
    artifacts_path = os.path.abspath(os.path.join(sacred_save_path, 'model_params.pkl'))
    with open(artifacts_path, 'rb') as f:
        ndgu_params = pickle.load(f)
    
    return ndgu_model, ndgu_params
