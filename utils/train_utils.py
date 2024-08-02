import os
import flax
import json
import jax.numpy as jnp
from matplotlib import cm
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

def setup_dirs(config):
    training_params = config.training_params
    paths = config.paths
    if paths.dir is None:
        config.paths.dir = os.path.join(
            os.curdir, 
            f'results/{training_params.net_name}/{config.system_name}/{config.trial_name}')
        paths.dir = config.paths.dir
    
    log_dir = os.path.join(paths.dir, 'log')
    plot_dir = os.path.join(paths.dir, 'plots')
    checkpoint_dir = os.path.join(paths.dir, 'checkpoint')
    checkpoint_dir = os.path.join(checkpoint_dir, 'best_model')
    checkpoint_dir = os.path.abspath(checkpoint_dir)

    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    dirs = {
        'log': log_dir,
        'plot': plot_dir,
        'ckpt': checkpoint_dir,
    }

    return dirs

def set_name(config):
    if 'mass_spring' in config.system_name:
        name = 'MassSpring'
    else:
        name = config.system_name
    return name

def create_net(name, net_params):
    if name == 'MassSpring':
        return MassSpringGNS(**net_params)
    else:
        return PHGNS2(**net_params)
    
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
    elif name == 'Alternator':
        return AlternatorGraphBuilder
    else:
        raise NotImplementedError(f'Graph builder not implemented for system {name}')
    
def random_batches(batch_size: int, min: int, max: int, rng: jax.Array):
    """ Return random permutation of jnp.arange(min, max) in batches of batch_size """
    steps_per_epoch = (max - min) // batch_size
    perms = jax.random.permutation(rng, max - min)
    perms = perms[: steps_per_epoch * batch_size].reshape(-1,batch_size)
    return perms

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

    cmap = cm.tab10

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

        layout = [['Q', 'Phi', 'H'],
                  ['Q_error', 'Phi_error','H_error']]
        ax = fig.subplot_mosaic(layout)

        ax['Q'].set_title('$Q$')
        ax['Q'].plot(ts, pred_data[0,:], color=cmap(0), ls='-', label='pred $Q_1$')
        ax['Q'].plot(ts, pred_data[2,:], color=cmap(1), ls='-', label='pred $Q_3$')
        ax['Q'].plot(ts, exp_data[0,:], color=cmap(0), ls='--', label='exp $Q_1$')
        ax['Q'].plot(ts, exp_data[2,:], color=cmap(1), ls='--', label='exp $Q_3$')
        ax['Q'].set_xlabel('Time [$s$]')
        ax['Q'].set_ylabel('$Q$')
        ax['Q'].legend()

        ax['Phi'].set_title('$\Phi$')
        ax['Phi'].plot(ts, pred_data[1,:], color=cmap(0), ls='-', label='pred $\Phi$')
        ax['Phi'].plot(ts, exp_data[1,:], color=cmap(0), ls='--', label='exp $\Phi$')
        ax['Phi'].set_xlabel('Time [$s$]')
        ax['Phi'].set_ylabel('$\Phi$')
        ax['Phi'].legend()

        ax['H'].set_title('Hamiltonian')
        ax['H'].plot(ts, pred_data[3,:], color=cmap(0), ls='-', label='predicted')
        ax['H'].plot(ts, exp_data[3,:], color=cmap(0), ls='--', label='expected')
        ax['H'].set_xlabel('Time [$s$]')
        ax['H'].set_ylabel('Hamiltonian')
        ax['H'].legend()
    
        ax['Q_error'].set_title('$Q$ Error')
        ax['Q_error'].plot(ts, exp_data[0,:] - pred_data[0,:], color=cmap(0), ls='-')
        ax['Q_error'].plot(ts, exp_data[2,:] - pred_data[2,:], color=cmap(1), ls='-')
        ax['Q_error'].set_xlabel('Time [$s$]')
        ax['Q_error'].set_ylabel('$Q$')

        ax['Phi_error'].set_title('$\Phi_1$ Error')
        ax['Phi_error'].plot(ts, exp_data[1,:] - pred_data[1,:], color=cmap(0), ls='-')
        ax['Phi_error'].set_xlabel('Time [$s$]')
        ax['Phi_error'].set_ylabel('$\Phi_1$')

        ax['H_error'].set_title('Hamiltonian Error')
        ax['H_error'].plot(ts, exp_data[3,:] - pred_data[3,:], color=cmap(0), ls='-')
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
        # fig.suptitle(f'{prefix}')

        layout = [['Q', 'Phi', 'H'],
                  ['Q_error', 'Phi_error', 'H_error']]
        ax = fig.subplot_mosaic(layout)

        ax['Q'].set_title('$Q$')
        ax['Q'].plot(ts, pred_data[0,:], color=cmap(0), ls='-', label='pred $Q_1$')
        ax['Q'].plot(ts, exp_data[0,:], color=cmap(0), ls='--', label='exp $Q_1$')
        ax['Q'].plot(ts, pred_data[2,:], color=cmap(1), ls='-', label='pred $Q_3$')
        ax['Q'].plot(ts, exp_data[2,:], color=cmap(1), ls='--', label='exp $Q_3$')
        ax['Q'].plot(ts, pred_data[3,:], color=cmap(2), ls='-', label='pred $Q_2$')
        ax['Q'].plot(ts, exp_data[3,:], color=cmap(2), ls='--', label='exp $Q_2$')
        ax['Q'].set_xlabel('Time [$s$]')
        ax['Q'].set_ylabel('$Q_1$')
        ax['Q'].legend(loc='upper right')

        ax['Phi'].set_title('$\Phi$')
        ax['Phi'].plot(ts, pred_data[1,:], color=cmap(0), ls='-', label='pred $\Phi_1$')
        ax['Phi'].plot(ts, exp_data[1,:], color=cmap(0), ls='--', label='exp $\Phi_1$')
        ax['Phi'].plot(ts, pred_data[4,:], color=cmap(1), ls='-', label='pred $\Phi_2$')
        ax['Phi'].plot(ts, exp_data[4,:], color=cmap(1), ls='--', label='exp $\Phi_2$')
        ax['Phi'].set_xlabel('Time [$s$]')
        ax['Phi'].set_ylabel('$\Phi$')
        ax['Phi'].legend(loc='upper right')

        ax['H'].set_title('Hamiltonian')
        ax['H'].plot(ts, pred_data[5,:], color=cmap(0), ls='-', label='pred H')
        ax['H'].plot(ts, exp_data[5,:], color=cmap(0), ls='--',label='exp H')
        ax['H'].set_xlabel('Time [$s$]')
        ax['H'].set_ylabel('Hamiltonian')
        ax['H'].legend(loc='upper right')
    
        ax['Q_error'].set_title('$Q$ Error')
        ax['Q_error'].plot(ts, exp_data[0,:] - pred_data[0,:], color=cmap(0), label='$Q_1$')
        ax['Q_error'].plot(ts, exp_data[2,:] - pred_data[2,:], color=cmap(1), label='$Q_3$')
        ax['Q_error'].plot(ts, exp_data[3,:] - pred_data[3,:], color=cmap(2), label='$Q_2$')
        ax['Q_error'].set_xlabel('Time [$s$]')
        ax['Q_error'].set_ylabel('$Q$')
        ax['Q_error'].legend(loc='upper right')

        ax['Phi_error'].set_title('$\Phi$ Error')
        ax['Phi_error'].plot(ts, exp_data[1,:] - pred_data[1,:], color=cmap(0), label='$\Phi_1$')
        ax['Phi_error'].plot(ts, exp_data[4,:] - pred_data[4,:], color=cmap(1),  label='$\Phi_2$')
        ax['Phi_error'].set_xlabel('Time [$s$]')
        ax['Phi_error'].set_ylabel('$\Phi$')
        ax['Phi_error'].legend(loc='upper right')

        ax['H_error'].set_title('Hamiltonian Error')
        ax['H_error'].plot(ts, exp_data[5,:] - pred_data[5,:], color=cmap(0))
        ax['H_error'].set_xlabel('Time [$s$]')
        ax['H_error'].set_ylabel('Hamiltonian')

        plt.savefig(os.path.join(plot_dir, f'{prefix}.png'))
        if show: plt.show()
        plt.close()
    elif system_name == 'Alternator':
        fig = plt.figure(layout="constrained", figsize=(20,10))
        # fig.suptitle(f'{prefix}')

        layout = [['Phi', 'p', 'theta', 'H'],
                  ['Phi_error', 'p_error', 'theta_error', 'H_error']]
        ax = fig.subplot_mosaic(layout)

        ax['Phi'].set_title('$\Phi$')
        ax['Phi'].plot(ts, pred_data[0,:].T, color=cmap(0), ls='-', label='pred $\phi_{Rf}$')
        ax['Phi'].plot(ts, exp_data[0,:].T, color=cmap(0), ls='--', label='exp $\phi_{Rf}$')
        ax['Phi'].plot(ts, pred_data[1,:].T, color=cmap(1), ls='-', label='pred $\phi_{Rkd}$')
        ax['Phi'].plot(ts, exp_data[1,:].T, color=cmap(1), ls='--', label='exp $\phi_{Rkd}$')
        ax['Phi'].plot(ts, pred_data[2,:].T, color=cmap(2), ls='-', label='pred $\phi_{Rkq}$')
        ax['Phi'].plot(ts, exp_data[2,:].T, color=cmap(2), ls='--', label='exp $\phi_{Rkq}$')
        ax['Phi'].plot(ts, pred_data[3,:].T, color=cmap(3), ls='-', label='pred $\phi_{Sa}$')
        ax['Phi'].plot(ts, exp_data[3,:].T, color=cmap(3), ls='--', label='exp $\phi_{Sa}$')
        ax['Phi'].plot(ts, pred_data[4,:].T, color=cmap(4), ls='-', label='pred $\phi_{Sb}$')
        ax['Phi'].plot(ts, exp_data[4,:].T, color=cmap(4), ls='--', label='exp $\phi_{Sb}$')
        ax['Phi'].plot(ts, pred_data[5,:].T, color=cmap(5), ls='-', label='pred $\phi_{Sc}$')
        ax['Phi'].plot(ts, exp_data[5,:].T, color=cmap(5), ls='--', label='exp $\phi_{Sc}$')
        ax['Phi'].set_xlabel('Time [$s$]')
        ax['Phi'].set_ylabel('$\Phi$')
        ax['Phi'].legend(loc='upper right')

        ax['p'].set_title('$p$')
        ax['p'].plot(ts, pred_data[6,:], color=cmap(0), ls='-', label='pred $p$')
        ax['p'].plot(ts, exp_data[6,:], color=cmap(0), ls='--', label='exp $p$')
        ax['p'].set_xlabel('Time [$s$]')
        ax['p'].set_ylabel('Rotor momentum $p$')
        ax['p'].legend(loc='upper right')

        ax['theta'].set_title(r'$\theta$')
        ax['theta'].plot(ts, pred_data[7,:], color=cmap(0), ls='-', label='pred $\theta$')
        ax['theta'].plot(ts, exp_data[7,:], color=cmap(0), ls='--', label='exp $\theta$')
        ax['theta'].set_xlabel('Time [$s$]')
        ax['theta'].set_ylabel(r'Rotor angle $\theta$')
        ax['theta'].legend(loc='upper right')

        ax['H'].set_title('Hamiltonian')
        ax['H'].plot(ts, pred_data[8,:], color=cmap(0), ls='-', label='predicted')
        ax['H'].plot(ts, exp_data[8,:], color=cmap(0), ls='--', label='expected')
        ax['H'].set_xlabel('Time [$s$]')
        ax['H'].set_ylabel('Hamiltonian')
        ax['H'].legend(loc='upper right')
    

        ax['Phi_error'].set_title('$\Phi$ Error')
        ax['Phi_error'].plot(ts, exp_data[0,:].T - pred_data[0,:].T, color=cmap(0), ls='-', label='exp $\phi_{Rf}$')
        ax['Phi_error'].plot(ts, exp_data[1,:].T - pred_data[1,:].T, color=cmap(1), ls='-', label='exp $\phi_{Rkd}$')
        ax['Phi_error'].plot(ts, exp_data[2,:].T - pred_data[2,:].T, color=cmap(2), ls='-', label='exp $\phi_{Rkq}$')
        ax['Phi_error'].plot(ts, exp_data[3,:].T - pred_data[3,:].T, color=cmap(3), ls='-', label='exp $\phi_{Sa}$')
        ax['Phi_error'].plot(ts, exp_data[4,:].T - pred_data[4,:].T, color=cmap(4), ls='-', label='exp $\phi_{Sb}$')
        ax['Phi_error'].plot(ts, exp_data[5,:].T - pred_data[5,:].T, color=cmap(5), ls='-', label='exp $\phi_{Sc}$')
        ax['Phi_error'].set_xlabel('Time [$s$]')
        ax['Phi_error'].set_ylabel('$\Phi$')
        ax['Phi_error'].legend(loc='upper right')

        ax['p_error'].set_title('$p$ Error')
        ax['p_error'].plot(ts, exp_data[6,:] - pred_data[6,:])
        ax['p_error'].set_xlabel('Time [$s$]')
        ax['p_error'].set_ylabel('Rotor momentum $p$')

        ax['theta_error'].set_title(r'$\theta$ Error')
        ax['theta_error'].plot(ts, exp_data[7,:] - pred_data[7,:], color=cmap(0), ls='-')
        ax['theta_error'].set_xlabel('Time [$s$]')
        ax['theta_error'].set_ylabel(r'$\theta$')

        ax['H_error'].set_title('Hamiltonian Error')
        ax['H_error'].plot(ts, exp_data[8,:] - pred_data[8,:], color=cmap(0), ls='-')
        ax['H_error'].set_xlabel('Time [$s$]')
        ax['H_error'].set_ylabel('Hamiltonian')

        fig.tight_layout()

        plt.savefig(os.path.join(plot_dir, f'{prefix}.png'))
        if show: plt.show()
        plt.close()

    plt.close()