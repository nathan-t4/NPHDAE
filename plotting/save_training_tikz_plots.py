import os, sys
sys.path.append('..')

import matplotlib.pyplot as plt
import tikzplotlib
import numpy as np

from common import load_metrics, tikzplotlib_fix_ncols, save_plot
import json

##### Experiment 1
exp_file_name = 'fhn/2024-11-12_18-27-53_fitz_hugh_nagano_less_data' # batch=128, pen_g=1e-2, lr=1e-4 cosine (same as dgu)
# exp_file_name = 'fhn_node/2024-11-12_18-36-14_train_node_fhn' # lr=1e-4 cosine, 100000, batch=128

##### Experiment 2
exp_file_name = 'dgu_simple/2024-11-17_16-43-40_baseline' # batch=128, pen_g=1e-2, lr=1e-4 cosine
"""
Batch size higher gives better results.
"""

tikz = True
name = "dgu_phndae"
n = 1000
T = 100000


sacred_save_path = os.path.abspath(os.path.join('../cyrus_experiments/runs/', exp_file_name, '1'))
metrics_file_str = os.path.abspath(os.path.join(sacred_save_path, 'metrics.json'))

with open(metrics_file_str, 'r') as f:
    results = json.load(f)

# Plot the training results
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
ax.plot(
    results['training.total_loss']['steps'][:T][::n], 
    results['training.total_loss']['values'][:T][::n], 
    label='Training Total Loss', 
)
ax.plot(
    results['testing.total_loss']['steps'][:T][::n],
    results['testing.total_loss']['values'][:T][::n],
    label='Testing Total Loss',
    linewidth=3,
)
ax.plot(
    results['training.data_loss']['steps'][:T][::n], 
    results['training.data_loss']['values'][:T][::n], 
    label='Training Data Loss', 
)
ax.plot(
    results['testing.data_loss']['steps'][:T][::n],
    results['testing.data_loss']['values'][:T][::n],
    label='Testing Data Loss',
    linewidth=3,
)
# g loss
ax.plot(
    results['training.g_loss']['steps'][:T][::n], 
    results['training.g_loss']['values'][:T][::n], 
    label='Training G Loss', 
)
ax.plot(
    results['testing.g_loss']['steps'][:T][::n],
    results['testing.g_loss']['values'][:T][::n],
    label='Testing G Loss',
    linewidth=3,
)
ax.set_yscale('log')
ax.grid()
ax.legend()
save_plot(fig, tikz, name)
plt.cla()
print('Min total loss', min(results['training.total_loss']['values']))
print('Min data loss', min(results['training.data_loss']['values']))
print('Min g loss', min(results['training.g_loss']['values']))