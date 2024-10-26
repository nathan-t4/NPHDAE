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
# exp_file_name = '2024-08-06_18-40-56_train_phdae_dgu'
# exp_file_name = '2024-09-25_13-44-35_train_phdae_dgu'
exp_file_name = '2024-09-23_19-45-01_train_phdae_dgu' # 1e-5
# exp_file_name = '2024-10-09_14-41-28_phdae_dgu_1e-5_scaled_g' # pen_g 1e-1, 50000 epochs
# exp_file_name = '2024-10-09_15-20-18_phdae_dgu_1e-5_5e-1'
exp_file_name = '2024-10-18_14-47-14_phdae_dgu_std' # user 1-1 NOTE Baseline
# exp_file_name = '2024-10-19_10-09-08_phdae_dgu_user51' # user 2.5-1, pen_g=1e-1 NOTE Better than 1
# exp_file_name = '2024-10-19_10-25-00_phdae_dgu_user51' # user 2.5-1, pen_g=1e-3 NOTE Not good.
# exp_file_name = '2024-10-19_10-35-46_phdae_dgu_user101' # user 10-1, pen_g=1e-1 NOTE Not as good as 2.5
# exp_file_name = '2024-10-19_11-22-52_phdae_dgu_user101' # user 5-1, pen_g=1e-1 NOTE Not as good as 2.5
# exp_file_name = '2024-10-19_16-53-09_phdae_dgu_user101' # user 2-1, pen_g=1e-2
# exp_file_name = '2024-10-19_17-35-02_phdae_dgu_user101' # user 2-1, pen_g=1e-2, 150000 epochs
# exp_file_name = '2024-10-19_22-42-55_phdae_dgu_user101' # user 2.5-1, pen_g=1e-2, 150000 epochs NOTE Not good
# exp_file_name = '2024-10-19_23-20-51_phdae_dgu_user101' # user 1-1, pen_g=1e-2, 150000 epochs
# exp_file_name = '2024-10-21_13-15-38_phdae_dgu_user101' # user 1-1, pen_g=1e-2, 100000 epochs
# exp_file_name = '2024-10-21_15-40-24_phdae_dgu_user51' # user 5-1, pen_g=1e-2, 100000 epochs
exp_file_name = '2024-10-21_16-35-23_phdae_dgu_user51' # user 10-1, pen_g=1e-2, 100000 epochs NOTE Not as good as 5-1, pen_g=1e-2
# exp_file_name = '2024-10-21_18-13-07_phdae_dgu_user51' # user 7-1, pen_g=1e-2, 150000 epochs 
# exp_file_name = '2024-10-21_19-35-53_phdae_dgu_user51' # user 5-1, pen_g=1e-2, 150000 epochs
exp_file_name = '2024-10-22_12-25-00_phdae_dgu_user51' # user 5-1, pen_g=1e-2, 200000 epochs
# exp_file_name = '2024-10-23_16-35-36_phdae_dgu_user51' # user 10-1, pen_g=1e-2, 200000 epochs
# exp_file_name = '2024-10-23_19-40-40_phdae_dgu_user51' # user 20-1, pen_g=1e-2, 200000 epochs
exp_file_name = '2024-10-23_20-44-49_phdae_dgu_user101' # user 10-1, pen_g=1e-2, 300000 epochs

sacred_save_path = os.path.abspath(os.path.join('../cyrus_experiments/runs/', exp_file_name, '1'))
metrics_file_str = os.path.abspath(os.path.join(sacred_save_path, 'metrics.json'))

with open(metrics_file_str, 'r') as f:
    results = json.load(f)

# Plot the training results
fig = plt.figure(figsize=(7.5,5))
ax = fig.add_subplot(111)
# for key in results.keys():
#     # if key == 'testing.loss': continue # Plot the testing loss last.
#     if 'training' in key: continue
#     if 'normalized' in key: continue
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
    linewidth=3,
)

ax.set_yscale('log')
ax.grid()
ax.set_title('Total loss')
ax.legend()
plt.savefig('training_curves_total.png')
plt.cla()

# Data loss
ax.plot(
    results['training.data_loss']['steps'], 
    results['training.data_loss']['values'], 
    label='training.data_loss', 
)
ax.plot(
    results['testing.data_loss']['steps'],
    results['testing.data_loss']['values'],
    label='testing.data_loss',
    linewidth=3,
)
ax.set_yscale('log')
ax.grid()
ax.legend()
ax.set_title('MSE on states')
plt.savefig('training_curves_data.png')
plt.cla()
# g loss
ax.plot(
    results['training.g_loss']['steps'], 
    results['training.g_loss']['values'], 
    label='training.g_loss', 
)
ax.plot(
    results['testing.g_loss']['steps'],
    results['testing.g_loss']['values'],
    label='testing.g_loss',
    linewidth=3,
)
ax.set_yscale('log')
ax.grid()
ax.legend()
ax.set_title('Constraint violation')
plt.savefig('training_curves_g.png')
plt.cla()
# ax.plot(
#     results['testing.data_loss']['steps'],
#     results['testing.data_loss']['values'],
#     label='testing.data_loss',
# )
