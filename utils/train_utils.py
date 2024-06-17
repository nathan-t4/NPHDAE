import os
import flax
import json
import jax.numpy as jnp
import matplotlib.pyplot as plt

from typing import Dict, Any
from functools import partial
from clu import metrics

from scripts.graph_builder import *
from scripts.graph_nets import *

def set_name(config):
    if 'mass_spring' in config.system_name:
        name = 'MassSpring'
    elif config.system_name == 'LC':
        name =  'LC'
    elif config.system_name == 'LC1':
        name = 'LC1'
    elif config.system_name == 'LC2':
        name = 'LC2'
    elif config.system_name == 'CoupledLC':
        name = 'CoupledLC'
    else:
        raise NotImplementedError()
    return name

def create_net(name, training_params, net_params):
    if training_params.net_name == 'GNS':
        if name == 'MassSpring':
            return MassSpringGNS(**net_params)
        elif name == 'LC1' or name == 'LC2':
            return LCGNS(**net_params)
        elif name == 'LC' or name == 'CoupledLC':
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        
    # elif training_params.net_name == 'GNODE':
    #     return GNODE(**net_params)
    
    else:
        raise RuntimeError('Invalid net name')
    
def create_graph_builder(name, training_params=None, net_params=None):
    if name == 'MassSpring':
        params = {
            'add_undirected_edges': training_params.add_undirected_edges, 
            'add_self_loops': training_params.add_self_loops,
            'vel_history': net_params.vel_history,
            'control_history': net_params.control_history,
        }
        return partial(MSDGraphBuilder, **params)
    elif name == 'LC':
        return LCGraphBuilder
    elif name == 'LC1':
        return LC1GraphBuilder
    elif name == 'LC2':
        return LC2GraphBuilder
    elif name == 'CoupledLC':
        return CoupledLCGraphBuilder

@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')

@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')

def add_prefix_to_keys(result: Dict[str, Any], prefix: str) -> Dict[str, Any]:
  """Adds a prefix to the keys of a dict, returning a new dict."""
  return {f'{prefix}_{key}': val for key, val in result.items()}

def save_evaluation_curves(dir: str, name: str, pred: jnp.ndarray, exp: jnp.ndarray) -> None:
    """ Save error plots from evaluation"""
    labels_fn = lambda xs, s: [f'{x} {s}' for x in xs] # helper function to create labels list
    assert pred.shape == exp.shape
    fig, ax = plt.subplots()
    ax.set_title(f'{name.capitalize()} Error')
    ax.set_xlabel('Time')

    ax.plot(jnp.arange(len(pred)), exp - pred, label=labels_fn(list(range(pred.shape[1])), 'error'))
    ax.legend()
    if not os.path.isdir(dir):
        os.makedirs(dir)
    plt.savefig(os.path.join(dir, f'{name}.png'))
    plt.close()

def save_params(work_dir, training_params, net_params):
    # Save run params to json
    run_params = {
        'training_params': training_params,
        'net_params': net_params
    }
    run_params_file = os.path.join(work_dir, 'run_params.js')
    with open(run_params_file, "w") as outfile:
        json.dump(run_params, outfile)