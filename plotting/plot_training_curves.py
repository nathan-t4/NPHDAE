import os, sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np

from common import load_metrics
import json

exp_file_name = '2024-08-05_01-58-32_train_mlp_rlc'
exp_file_name = '2024-08-05_09-56-07_train_phdae_rlc'

sacred_save_path = os.path.abspath(os.path.join('../cyrus_experiments/runs/', exp_file_name, '1'))
metrics_file_str = os.path.abspath(os.path.join(sacred_save_path, 'metrics.json'))

with open(metrics_file_str, 'r') as f:
    results = json.load(f)

# Plot the training results
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
# for key in results.keys():
#     if key == 'testing.loss': continue # Plot the testing loss last.
#     ax.plot(results[key]['steps'], results[key]['values'], label=key)
ax.plot(
    results['training.total_loss']['steps'], 
    results['training.total_loss']['values'], 
    label='training.total_loss', 
)
ax.plot(
    results['testing.total_loss']['steps'], 
    results['testing.total_loss']['values'], 
    label='testing.total_loss', 
)
ax.set_yscale('log')
ax.grid()
ax.legend()
plt.savefig('training_curves.png')