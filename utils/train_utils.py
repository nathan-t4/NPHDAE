import os
import flax
import json
import jax.numpy as jnp
import matplotlib.pyplot as plt

from typing import Dict, Any
from clu import metrics

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