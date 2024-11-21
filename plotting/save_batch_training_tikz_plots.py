import os, sys
sys.path.append('..')

import matplotlib.pyplot as plt
import tikzplotlib
import numpy as np

from matplotlib.colors import TABLEAU_COLORS as tableau

from common import load_metrics, tikzplotlib_fix_ncols, save_plot
import json


batch_exp_dirs = ('batch_training/fhn', 'batch_training/fhn_node')
trial_names = ("baseline", "train_node_fhn")
labels = ("PHNDAE", "NODE")


tikz = False
fig_name = "experiment_1_training"
n = 200
T = int(1e5)
alpha = 0.3
num_trials = 5

colors = list(tableau.values())


fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)

def initialize_empty_batch_results():
    batch_results = {
        'training.total_loss': {'steps': [], 'values': []},
        'training.data_loss': {'steps': [], 'values': []},
        'training.g_loss': {'steps': [], 'values': []},
        'testing.total_loss': {'steps': [], 'values': []},
        'testing.data_loss': {'steps': [], 'values': []},
        'testing.g_loss': {'steps': [], 'values': []},
    }
    return batch_results

def append_data_to_dict(batch_exp_dir, batch_results:dict, num_trials:int, trial_name='baseline'):
    batch_abs_path = os.path.abspath(os.path.join('../cyrus_experiments/runs/', batch_exp_dir))
    for i in range(num_trials):
        sacred_save_path = os.path.abspath(os.path.join(batch_abs_path, f'{i}_{trial_name}', '1'))
        metrics_file_str = os.path.abspath(os.path.join(sacred_save_path, 'metrics.json'))

        with open(metrics_file_str, 'r') as f:
            results = json.load(f)

        batch_results['training.total_loss']['steps'].append(
            results['training.total_loss']['steps'][:T][::n]
        )
        batch_results['training.total_loss']['values'].append(
            results['training.total_loss']['values'][:T][::n], 
        )

        batch_results['testing.total_loss']['steps'].append(
            results['testing.total_loss']['steps'][:T][::n]
        )
        batch_results['testing.total_loss']['values'].append(
            results['testing.total_loss']['values'][:T][::n],
        )

        batch_results['training.data_loss']['steps'].append(
            results['training.data_loss']['steps'][:T][::n]
        )
        batch_results['training.data_loss']['values'].append(
            results['training.data_loss']['values'][:T][::n],
        )

        batch_results['testing.data_loss']['steps'].append(
            results['testing.data_loss']['steps'][:T][::n]
        )
        batch_results['testing.data_loss']['values'].append(
            results['testing.data_loss']['values'][:T][::n],
        )

        try:
            # Training for baselines do not have g loss
            batch_results['training.g_loss']['steps'].append(
                results['training.g_loss']['steps'][:T][::n]
            )
            batch_results['training.g_loss']['values'].append(
                results['training.g_loss']['values'][:T][::n],
            )

            batch_results['testing.g_loss']['steps'].append(
                results['testing.g_loss']['steps'][:T][::n]
            )
            batch_results['testing.g_loss']['values'].append(
                results['testing.g_loss']['values'][:T][::n],
            )
        except:
            pass
        
    return batch_results

def get_percentiles(data, axis):
    data_25 = np.percentile(data, 25, axis=axis)
    data_50 = np.percentile(data, 50, axis=axis)
    data_75 = np.percentile(data, 75, axis=axis)
    return data_25, data_50, data_75

def plot_percentile_curves(ax, batch_results, key, color, alpha, label, axis=0):
    T = batch_results[key]['steps'][0]
    data_25, data_50, data_75 = get_percentiles(batch_results[key]['values'], axis=axis)
    ax.plot(
        T,
        data_25,
        color=color,
        alpha=alpha,
    )
    
    ax.plot(
        T,
        data_50,
        color=color,
        # linewidth=3,
        label=label,
    )

    ax.plot(
        T,
        data_75,
        color=color,
        alpha=alpha,
    )

    ax.fill_between(
        T,
        data_25,
        data_75,
        color=color,
        alpha=alpha,
    )
    return ax


color_idx = 0
for i, (exp_dir, name) in enumerate(zip(batch_exp_dirs, trial_names)):
    batch_results = initialize_empty_batch_results()
    batch_results = append_data_to_dict(exp_dir, batch_results, num_trials, name)

    # Training total loss
    # total_loss_color = '#ff7f0e'
    # ax = plot_percentile_curves(ax, batch_results, 'training.total_loss', color=total_loss_color, alpha=alpha, label='Total Loss')


    # Training data loss
    # data_loss_color = '#1f77b4'
    ax = plot_percentile_curves(ax, batch_results, 'training.data_loss', color=colors[color_idx], alpha=alpha, label=f'{labels[i]} MSE')


    # # Training g loss
    # g_loss_color = '#2ca02c'
    # ax = plot_percentile_curves(ax, batch_results, 'training.g_loss', color=g_loss_color, alpha=alpha, label='$g$ MSE')
    
    color_idx = color_idx + 1

ax.set_yscale('log')
ax.grid()
ax.legend()
ax.set_ylim(top=1.0)
save_plot(fig, tikz, fig_name)
plt.cla()