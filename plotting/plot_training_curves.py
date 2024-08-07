import os, sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np

from common import load_metrics
import json

exp_file_name = '2024-08-06_12-03-04_train_mlp_rlc'
# exp_file_name = '2024-08-06_12-00-44_train_node_rlc'
# exp_file_name = '2024-08-06_11-18-03_train_phdae_rlc'
exp_file_name = '2024-08-06_16-41-37_train_phdae_rlc'
exp_file_name = '2024-08-06_18-40-56_train_phdae_dgu'

sacred_save_path = os.path.abspath(os.path.join('../cyrus_experiments/runs/', exp_file_name, '1'))
metrics_file_str = os.path.abspath(os.path.join(sacred_save_path, 'metrics.json'))

with open(metrics_file_str, 'r') as f:
    results = json.load(f)

# Plot the training results
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
for key in results.keys():
    # if key == 'testing.loss': continue # Plot the testing loss last.
    if 'training' in key: continue
    if 'normalized' in key: continue
    ax.plot(results[key]['steps'], results[key]['values'], label=key)
ax.plot(
    results['testing.total_loss']['steps'], 
    results['testing.total_loss']['values'], 
    label='testing.total_loss', 
)
# ax.plot(
#     results['testing.g_loss']['steps'],
#     results['testing.g_loss']['values'],
#     label='testing.g_loss',
# )
# ax.plot(
#     results['testing.data_loss']['steps'],
#     results['testing.data_loss']['values'],
#     label='testing.data_loss',
# )
ax.set_yscale('log')
ax.grid()
ax.legend()
plt.savefig('training_curves.png')