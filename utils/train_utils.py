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

@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')

@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')

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
        elif name == 'LC1' or name == 'LC2' or 'CoupledLC':
            return LCGNS(**net_params)
        elif name == 'LC':
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

def plot_evaluation_curves(
        ts, pred_data, exp_data, aux_data, prefix, plot_dir, show=False
    ):
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    system_name = aux_data['name']
    if system_name == 'MassSpring':
        m = jnp.round(aux_data[0], 3)
        k = jnp.round(aux_data[1], 3)
        b = jnp.round(aux_data[2], 3)

        q0 = jnp.round(exp_data[0,0], 3)
        a0 = jnp.round(exp_data[1,0], 3)

        N = len(m)
        # title = f"{prefix}: Mass {i} \n $m_{i}$ = " + "{:.2f},".format(m[i]) + f" $k_{i}$ = " + "{:.2f},".format(k[i]) + f" $b_{i}$ = " + "{:.2f}".format(b[i])
        fig = plt.figure(layout="constrained", figsize=(20,10))
        fig.suptitle(f'{prefix}')
    
        layout = []
        for i in range(N):
            layout.append([f'q{i}', f'qdd{i}', f'q{i}_error', f'qdd{i}_error'])
        ax = fig.subplot_mosaic(layout)

        for i in range(N):
            ax[f'q{i}'].set_title(f'{i} Position')
            ax[f'q{i}'].plot(ts, pred_data[0,:,i], label='predicted')
            ax[f'q{i}'].plot(ts, exp_data[0,:,i], label='expected')
            ax[f'q{i}'].set_xlabel('Time [$s$]')
            ax[f'q{i}'].set_ylabel('Position [$m$]')
            ax[f'q{i}'].legend()

            ax[f'qdd{i}'].set_title(f'{i} Acceleration')
            ax[f'qdd{i}'].plot(ts, pred_data[1,:,i], label='predicted')
            ax[f'qdd{i}'].plot(ts, exp_data[1,:,i], label='expected')
            ax[f'qdd{i}'].set_xlabel('Time [$s$]')
            ax[f'qdd{i}'].set_ylabel(r'Acceleration [$\mu m/s^2$]')
            ax[f'qdd{i}'].legend()
        
            ax[f'q{i}_error'].set_title(f'{i} Position Error')
            ax[f'q{i}_error'].plot(ts, exp_data[0,:,i] - pred_data[0,:,i])
            ax[f'q{i}_error'].set_xlabel('Time [$s$]')
            ax[f'q{i}_error'].set_ylabel('Position [$m$]')

            ax[f'qdd{i}_error'].set_title(f'{i} Acceleration Error')
            ax[f'qdd{i}_error'].plot(ts, exp_data[1,:,i] - pred_data[1,:,i])
            ax[f'qdd{i}_error'].set_xlabel('Time [$s$]')
            ax[f'qdd{i}_error'].set_ylabel(r'Acceleration [$\mu m/s^2$]')

        plt.savefig(os.path.join(plot_dir, f'{prefix}.png'))
        if show: plt.show()
        plt.close()
    
    elif system_name == 'LC':
        fig = plt.figure(layout="constrained", figsize=(20,10))
        fig.suptitle(f'{prefix}')

        layout = [['Q', 'Phi', 'H'],
                  ['Q_error', 'Phi_error', 'H_error']]
        ax = fig.subplot_mosaic(layout)

        ax['Q'].set_title('Q')
        ax['Q'].plot(ts, pred_data[0,:], label='predicted')
        ax['Q'].plot(ts, exp_data[0,:], label='expected')
        ax['Q'].set_xlabel('Time [$s$]')
        ax['Q'].set_ylabel('Q')
        ax['Q'].legend()

        ax['Phi'].set_title('Phi')
        ax['Phi'].plot(ts, pred_data[1,:], label='predicted')
        ax['Phi'].plot(ts, exp_data[1,:], label='expected')
        ax['Phi'].set_xlabel('Time [$s$]')
        ax['Phi'].set_ylabel('Phi')
        ax['Phi'].legend()

        ax['H'].set_title('Hamiltonian')
        ax['H'].plot(ts, pred_data[2,:], label='predicted')
        ax['H'].plot(ts, exp_data[2,:], label='expected')
        ax['H'].set_xlabel('Time [$s$]')
        ax['H'].set_ylabel('Hamiltonian')
        ax['H'].legend()
    
        ax['Q_error'].set_title('Q Error')
        ax['Q_error'].plot(ts, exp_data[0,:] - pred_data[0,:])
        ax['Q_error'].set_xlabel('Time [$s$]')
        ax['Q_error'].set_ylabel('Q')

        ax['Phi_error'].set_title('Phi Error')
        ax['Phi_error'].plot(ts, exp_data[1,:] - pred_data[1,:])
        ax['Phi_error'].set_xlabel('Time [$s$]')
        ax['Phi_error'].set_ylabel('Phi')

        ax['H_error'].set_title('Hamiltonian Error')
        ax['H_error'].plot(ts, exp_data[2,:] - pred_data[2,:])
        ax['H_error'].set_xlabel('Time [$s$]')
        ax['H_error'].set_ylabel('Hamiltonian')

        plt.savefig(os.path.join(plot_dir, f'{prefix}.png'))
        if show: plt.show()
        plt.close()

    elif system_name == 'LC1':
        fig = plt.figure(layout="constrained", figsize=(20,10))
        fig.suptitle(f'{prefix}')

        layout = [['Q1', 'Phi1', 'Q3', 'H'],
                  ['Q1_error', 'Phi1_error', 'Q3_error', 'H_error']]
        ax = fig.subplot_mosaic(layout)

        ax['Q1'].set_title('$Q_1$')
        ax['Q1'].plot(ts, pred_data[0,:], label='predicted')
        ax['Q1'].plot(ts, exp_data[0,:], label='expected')
        ax['Q1'].set_xlabel('Time [$s$]')
        ax['Q1'].set_ylabel('$Q_1$')
        ax['Q1'].legend()

        ax['Phi1'].set_title('$\Phi_1$')
        ax['Phi1'].plot(ts, pred_data[1,:], label='predicted')
        ax['Phi1'].plot(ts, exp_data[1,:], label='expected')
        ax['Phi1'].set_xlabel('Time [$s$]')
        ax['Phi1'].set_ylabel('$\Phi_1$')
        ax['Phi1'].legend()

        ax['Q3'].set_title('$Q_3$')
        ax['Q3'].plot(ts, pred_data[2,:], label='predicted')
        ax['Q3'].plot(ts, exp_data[2,:], label='expected')
        ax['Q3'].set_xlabel('Time [$s$]')
        ax['Q3'].set_ylabel('$Q_3$')
        ax['Q3'].legend()

        ax['H'].set_title('Hamiltonian')
        ax['H'].plot(ts, pred_data[3,:], label='predicted')
        ax['H'].plot(ts, exp_data[3,:], label='expected')
        ax['H'].set_xlabel('Time [$s$]')
        ax['H'].set_ylabel('Hamiltonian')
        ax['H'].legend()
    
        ax['Q1_error'].set_title('$Q_1$ Error')
        ax['Q1_error'].plot(ts, exp_data[0,:] - pred_data[0,:])
        ax['Q1_error'].set_xlabel('Time [$s$]')
        ax['Q1_error'].set_ylabel('$Q_1$')

        ax['Phi1_error'].set_title('$\Phi_1$ Error')
        ax['Phi1_error'].plot(ts, exp_data[1,:] - pred_data[1,:])
        ax['Phi1_error'].set_xlabel('Time [$s$]')
        ax['Phi1_error'].set_ylabel('$\Phi_1$')

        ax['Q3_error'].set_title('$Q_3$ Error')
        ax['Q3_error'].plot(ts, exp_data[2,:] - pred_data[2,:])
        ax['Q3_error'].set_xlabel('Time [$s$]')
        ax['Q3_error'].set_ylabel('$Q_3$')

        ax['H_error'].set_title('Hamiltonian Error')
        ax['H_error'].plot(ts, exp_data[3,:] - pred_data[3,:])
        ax['H_error'].set_xlabel('Time [$s$]')
        ax['H_error'].set_ylabel('Hamiltonian')

        plt.savefig(os.path.join(plot_dir, f'{prefix}.png'))
        if show: plt.show()
        plt.close()

    elif system_name == 'LC2':
        fig = plt.figure(layout="constrained", figsize=(20,10))
        fig.suptitle(f'{prefix}')

        layout = [['Q2', 'Phi2', 'H'],
                  ['Q2_error', 'Phi2_error', 'H_error']]
        ax = fig.subplot_mosaic(layout)

        ax['Q2'].set_title('$Q_2$')
        ax['Q2'].plot(ts, pred_data[0,:], label='predicted')
        ax['Q2'].plot(ts, exp_data[0,:], label='expected')
        ax['Q2'].set_xlabel('Time [$s$]')
        ax['Q2'].set_ylabel('$Q_2$')
        ax['Q2'].legend()

        ax['Phi2'].set_title('$\Phi_2$')
        ax['Phi2'].plot(ts, pred_data[1,:], label='predicted')
        ax['Phi2'].plot(ts, exp_data[1,:], label='expected')
        ax['Phi2'].set_xlabel('Time [$s$]')
        ax['Phi2'].set_ylabel('$\Phi_2$')
        ax['Phi2'].legend()

        ax['H'].set_title('Hamiltonian')
        ax['H'].plot(ts, pred_data[2,:], label='predicted')
        ax['H'].plot(ts, exp_data[2,:], label='expected')
        ax['H'].set_xlabel('Time [$s$]')
        ax['H'].set_ylabel('Hamiltonian')
        ax['H'].legend()
    
        ax['Q2_error'].set_title('$Q_2$ Error')
        ax['Q2_error'].plot(ts, exp_data[0,:] - pred_data[0,:])
        ax['Q2_error'].set_xlabel('Time [$s$]')
        ax['Q2_error'].set_ylabel('$Q_2$')

        ax['Phi2_error'].set_title('$\Phi_2$ Error')
        ax['Phi2_error'].plot(ts, exp_data[1,:] - pred_data[1,:])
        ax['Phi2_error'].set_xlabel('Time [$s$]')
        ax['Phi2_error'].set_ylabel('$\Phi_2$')

        ax['H_error'].set_title('Hamiltonian Error')
        ax['H_error'].plot(ts, exp_data[2,:] - pred_data[2,:])
        ax['H_error'].set_xlabel('Time [$s$]')
        ax['H_error'].set_ylabel('Hamiltonian')

        plt.savefig(os.path.join(plot_dir, f'{prefix}.png'))
        if show: plt.show()
        plt.close()
    
    elif system_name == 'CoupledLC' or system_name == 'CompLCCircuits':
        fig = plt.figure(layout="constrained", figsize=(20,10))
        fig.suptitle(f'{prefix}')

        layout = [['Q1', 'Phi1', 'Q3', 'Q2', 'Phi2', 'H'],
                  ['Q1_error', 'Phi1_error', 'Q3_error', 'Q2_error', 'Phi2_error', 'H_error']]
        ax = fig.subplot_mosaic(layout)

        ax['Q1'].set_title('$Q_1$')
        ax['Q1'].plot(ts, pred_data[0,:], label='predicted')
        ax['Q1'].plot(ts, exp_data[0,:], label='expected')
        ax['Q1'].set_xlabel('Time [$s$]')
        ax['Q1'].set_ylabel('$Q_1$')
        ax['Q1'].legend()

        ax['Phi1'].set_title('$\Phi_1$')
        ax['Phi1'].plot(ts, pred_data[1,:], label='predicted')
        ax['Phi1'].plot(ts, exp_data[1,:], label='expected')
        ax['Phi1'].set_xlabel('Time [$s$]')
        ax['Phi1'].set_ylabel('$\Phi_1$')
        ax['Phi1'].legend()

        ax['Q3'].set_title('$Q_3$')
        ax['Q3'].plot(ts, pred_data[2,:], label='predicted')
        ax['Q3'].plot(ts, exp_data[2,:], label='expected')
        ax['Q3'].set_xlabel('Time [$s$]')
        ax['Q3'].set_ylabel('$Q_3$')
        ax['Q3'].legend()

        ax['Q2'].set_title('$Q_2$')
        ax['Q2'].plot(ts, pred_data[3,:], label='predicted')
        ax['Q2'].plot(ts, exp_data[3,:], label='expected')
        ax['Q2'].set_xlabel('Time [$s$]')
        ax['Q2'].set_ylabel('$Q_2$')
        ax['Q2'].legend()


        ax['Phi2'].set_title('$\Phi_2$')
        ax['Phi2'].plot(ts, pred_data[4,:], label='predicted')
        ax['Phi2'].plot(ts, exp_data[4,:], label='expected')
        ax['Phi2'].set_xlabel('Time [$s$]')
        ax['Phi2'].set_ylabel('$\Phi_2$')
        ax['Phi2'].legend()

        ax['H'].set_title('Hamiltonian')
        ax['H'].plot(ts, pred_data[5,:], label='predicted')
        ax['H'].plot(ts, exp_data[5,:], label='expected')
        ax['H'].set_xlabel('Time [$s$]')
        ax['H'].set_ylabel('Hamiltonian')
        ax['H'].legend()
    
        ax['Q1_error'].set_title('$Q_1$ Error')
        ax['Q1_error'].plot(ts, exp_data[0,:] - pred_data[0,:])
        ax['Q1_error'].set_xlabel('Time [$s$]')
        ax['Q1_error'].set_ylabel('$Q_1$')

        ax['Phi1_error'].set_title('$\Phi_1$ Error')
        ax['Phi1_error'].plot(ts, exp_data[1,:] - pred_data[1,:])
        ax['Phi1_error'].set_xlabel('Time [$s$]')
        ax['Phi1_error'].set_ylabel('$\Phi_1$')

        ax['Q3_error'].set_title('$Q_3$ Error')
        ax['Q3_error'].plot(ts, exp_data[2,:] - pred_data[2,:])
        ax['Q3_error'].set_xlabel('Time [$s$]')
        ax['Q3_error'].set_ylabel('$Q_3$')

        ax['Q2_error'].set_title('$Q_2$ Error')
        ax['Q2_error'].plot(ts, exp_data[3,:] - pred_data[3,:])
        ax['Q2_error'].set_xlabel('Time [$s$]')
        ax['Q2_error'].set_ylabel('$Q_2$')

        ax['Phi2_error'].set_title('$\Phi_2$ Error')
        ax['Phi2_error'].plot(ts, exp_data[4,:] - pred_data[4,:])
        ax['Phi2_error'].set_xlabel('Time [$s$]')
        ax['Phi2_error'].set_ylabel('$\Phi_2$')

        ax['H_error'].set_title('Hamiltonian Error')
        ax['H_error'].plot(ts, exp_data[5,:] - pred_data[5,:])
        ax['H_error'].set_xlabel('Time [$s$]')
        ax['H_error'].set_ylabel('Hamiltonian')

        plt.savefig(os.path.join(plot_dir, f'{prefix}.png'))
        if show: plt.show()
        plt.close()    
    plt.close()