import os
import flax
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

def plot_evaluation_curves(
        ts, pred_data, exp_data, aux_data, prefix, plot_dir, prediction='acceleration', show=False
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

        layout = [['Q', 'Phi', 'Q_error', 'Phi_error']]
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
    
        ax['Q_error'].set_title('Q Error')
        ax['Q_error'].plot(ts, exp_data[0,:] - pred_data[0,:])
        ax['Q_error'].set_xlabel('Time [$s$]')
        ax['Q_error'].set_ylabel('Q')

        ax['Phi_error'].set_title('Phi Error')
        ax['Phi_error'].plot(ts, exp_data[1,:] - pred_data[1,:])
        ax['Phi_error'].set_xlabel('Time [$s$]')
        ax['Phi_error'].set_ylabel('Phi')

        plt.savefig(os.path.join(plot_dir, f'{prefix}.png'))
        if show: plt.show()
        plt.close()
       
        
    plt.close()