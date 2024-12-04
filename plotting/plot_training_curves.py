import os, sys
sys.path.append('..')

import matplotlib.pyplot as plt
import tikzplotlib
import numpy as np

from common import load_metrics
import json

exp_file_name = '2024-08-06_12-03-04_train_mlp_rlc'
# exp_file_name = '2024-08-06_12-00-44_train_node_rlc'
# exp_file_name = '2024-08-06_11-18-03_train_phdae_rlc'
exp_file_name = '2024-08-06_16-41-37_train_phdae_rlc'


exp_file_name = 'dgu/2024-11-08_10-23-19_phdae_dgu_user_1' # slightly less data NOTE BEST
exp_file_name = 'dgu/2024-11-09_11-27-57_phdae_dgu_user_1' # batch=512,lr=1e-4,hidden_size=100
exp_file_name = 'dgu/2024-11-09_12-05-45_phdae_dgu_user_1' # batch=512,lr=1e-5,hidden_size=100
exp_file_name = 'dgu/2024-11-09_18-41-06_phdae_dgu_user_1' # batch=128, lr=1e-5, hidden_size=100
exp_file_name = 'dgu/2024-11-09_21-42-12_phdae_dgu_user_1' # batch=512, lr=1e-5, hidden_size=100, less data NOTE same params as BEST but less data
exp_file_name = 'dgu/2024-11-10_08-00-38_phdae_dgu_user_1' # batch=128, lr=1e-5, hidden_size=100, scaling 10, less data
# exp_file_name = 'dgu/2024-11-10_08-01-34_phdae_dgu_user_1' # batch=128, lr=1e-5, hidden_size=100, scaling 1, less data
exp_file_name = 'dgu/2024-11-10_08-49-37_phdae_dgu_user_1' # batch=128, lr=1e-5, hidden_size=100, pen_g=1e-2, scaling 1, less data
exp_file_name = 'dgu/2024-11-10_08-50-38_phdae_dgu_user_1' # batch=256, lr=1e-5, hidden_size=100, pen_g=1e-1, scaling 1, less data
exp_file_name = 'dgu/2024-11-08_10-23-19_phdae_dgu_user_1' 
exp_file_name = 'dgu/2024-11-10_19-22-23_phdae_dgu_user_1' # batch=256, pen_g=1e-1, lr=1e-5, more data
# exp_file_name = 'dgu/2024-11-10_19-24-12_phdae_dgu_user_1' # batch=256, pen_g=1e-1, lr=1e-4, more data
# TODO
exp_file_name = 'dgu/2024-11-11_09-30-54_phdae_dgu_user_1' # batch=256, pen_g=1e-1, lr=1e-6 NOTE this one is good...
# exp_file_name = 'dgu/2024-11-11_09-58-28_phdae_dgu_user_1' # batch=128, pen_g=1e-1, lr=1e-6
# exp_file_name = 'dgu/2024-11-11_15-31-37_phdae_dgu_user_1' # batch=512, pen_g=1e-1, lr=1e-6
exp_file_name = 'dgu/2024-11-11_22-25-57_phdae_dgu_cosine' # batch=256, pen_g=1e-1, optax.schedules.cosine_decay_schedule(1e-5,5e5)
exp_file_name = 'dgu/2024-11-12_11-46-47_phdae_dgu_cosine' # batch=256, pen_g=1e-1, lr=1e-5, less data, scaled voltage
# exp_file_name = 'dgu/2024-11-12_12-31-30_phdae_dgu_cosine' # batch=256, pen_g=1e-2, lr=1e-5, less data, scaled voltage
# exp_file_name = 'dgu/2024-11-12_12-32-50_phdae_dgu_cosine' # batch=256, pen_g=1e-1, lr=1e-4 cosine, less data, scaled voltage
# exp_file_name = 'dgu/2024-11-12_12-46-59_phdae_dgu_cosine' # batch=64, pen_g=1e-1, lr=1e-5, less data, scaled voltage
# exp_file_name = 'dgu/2024-11-12_12-49-17_phdae_dgu_cosine' # batch=512, pen_g=1e-1, lr=1e-5, less data, scaled voltage
# exp_file_name = 'dgu/2024-11-12_13-23-34_phdae_dgu_less_data' # batch=512, pen_g=1e-1, lr=1e-5, less data, scaled voltage
# exp_file_name = 'dgu/2024-11-12_13-31-53_phdae_dgu_less_data' # batch=64, pen_g=1e-1, lr=1e-6, less data, scaled voltage
exp_file_name = 'dgu/2024-11-12_13-36-59_phdae_dgu_less_data' # batch=128, pen_g=1e-1, lr=1e-4
exp_file_name = 'dgu/2024-11-12_15-35-29_phdae_dgu_less_data' # batch=64, pen_g=1e-1, lr=1e-5
# exp_file_name = 'dgu/2024-11-12_15-56-00_phdae_dgu_less_data' # batch=128, pen_g=1e-1, lr=1e-5, scaling=5

# TODO
# exp_file_name = 'dgu/2024-11-12_16-23-56_phdae_dgu_less_data' # batch=128, pen_g=1e-1, lr=1e-4, 
# exp_file_name = 'dgu/2024-11-12_16-36-18_phdae_dgu_less_data' # batch=256, pen_g=1e-1, lr=1e-4
exp_file_name = 'dgu/2024-11-12_16-37-04_phdae_dgu_less_data' # batch=128, pen_g=1e-2, lr=1e-4
# exp_file_name = 'dgu/2024-11-12_16-38-10_phdae_dgu_less_data' # batch=128, pen_g=1e-1, lr=1e-4 cosine
# exp_file_name = 'dgu/2024-11-12_16-56-53_phdae_dgu_less_data' # batch=256, pen_g=1e-1, lr=1e-4 cosine
# exp_file_name = 'dgu/2024-11-12_16-58-02_phdae_dgu_less_data' # batch=128, pen_g=1e-1, lr=3e-4 cosine
# exp_file_name = 'dgu/2024-11-12_17-00-11_phdae_dgu_less_data' # batch=128, pen_g=1e-2, lr=1e-4 cosine
# exp_file_name = 'dgu/2024-11-12_20-18-23_phdae_dgu_less_data' # batch=64, pen_g=1e-2, lr=1e-4 cosine, 300000 epochs # NOTE BEST
# exp_file_name = 'dgu/2024-11-12_21-33-25_phdae_dgu' # batch=64, pen_g=1e-2, lr=1e-4 cosine

exp_file_name = 'dgu/2024-11-13_12-03-20_phdae_dgu' # batch=128, pen_g=1.0, lr=1e-4 cosine
exp_file_name = 'dgu/2024-11-13_12-02-03_phdae_dgu' # batch=128, pen_g=1e-1, lr=1e-4 cosine
exp_file_name = 'dgu/2024-11-13_10-03-57_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-4 cosine
exp_file_name = 'dgu/2024-11-13_10-08-08_phdae_dgu' # batch=128, pen_g=1e-3, lr=1e-4 cosine
exp_file_name = 'dgu/2024-11-13_10-07-04_phdae_dgu' # batch=128, pen_g=1e-4, lr=1e-4 cosine
exp_file_name = 'dgu/2024-11-13_10-03-00_phdae_dgu' # batch=128, pen_g=0.0, lr=1e-4 cosine
# pen_g=1e-2 is best
exp_file_name = 'dgu/2024-11-12_19-29-45_phdae_dgu_less_data' # batch=256, pen_g=1e-2, lr=1e-4 cosine
exp_file_name = 'dgu/2024-11-12_19-30-39_phdae_dgu_less_data' # batch=64, pen_g=1e-2, lr=1e-4 cosine
# exp_file_name = 'dgu/2024-11-13_12-53-14_phdae_dgu' # batch=32, pen_g=1e-2, lr=1e-4 cosine

# exp_file_name = 'dgu/2024-11-13_12-54-54_phdae_dgu' # batch=64, pen_g=1e-2, lr=1e-5 cosine

# exp_file_name = 'dgu/2024-11-13_13-46-55_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-4 cosine, train gradH directly
# exp_file_name = 'dgu/2024-11-13_20-12-23_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-5 cosine, train gradH directly
# exp_file_name = 'dgu/2024-11-13_20-29-55_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-5 cosine, train gradH directly, more data
# exp_file_name = 'dgu/2024-11-13_20-30-36_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-4 cosine, train gradH directly, more data
exp_file_name = 'dgu/2024-11-13_20-32-22_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-4 cosine, train gradH directly, scale=10 NOTE BEST
# exp_file_name = 'dgu/2024-11-13_20-33-12_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-4 cosine, train gradH directly, scale=5
# exp_file_name = 'dgu/2024-11-13_21-01-27_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-4 cosine, train gradH irectly, scale=50
# exp_file_name = 'dgu/2024-11-13_21-01-58_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-5 cosine, train gradH irectly, scale=10
# # exp_file_name = 'dgu/2024-11-13_21-08-24_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-4 cosine scale=10
# exp_file_name = 'dgu/2024-11-13_21-09-12_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-4 cosine scale=50
# # exp_file_name = 'dgu/2024-11-13_23-43-15_phdae_dgu' #  batch=64, pen_g=1e-2, lr=1e-4 cosine scale=50
# # exp_file_name = 'dgu/2024-11-13_23-44-26_phdae_dgu'  #  batch=128, pen_g=1e-2, lr=1e-4 cosine scale=50, 3e5 epochs
# # exp_file_name = 'dgu/2024-11-14_11-38-38_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-4 cosine scale=50, normal voltage
# # exp_file_name = 'dgu/2024-11-14_12-33-45_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-4 cosine scale=5000, normal voltage
# exp_file_name = 'dgu/2024-11-14_13-58-27_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-4 cosine scale=50000, normal voltage
# exp_file_name = 'dgu/2024-11-14_14-02-53_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-4 cosine, scale=[0.1,0.1,50000,50000,50000,0.1]
# exp_file_name = 'dgu/2024-11-14_14-07-47_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-4 cosine, 'scalings': [100,50,50000,50000,50000,50],
# exp_file_name = 'dgu/2024-11-14_14-19-28_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-3 cosine, 'scalings': [100,50,50000,50000,50000,50],
# exp_file_name = 'dgu/2024-11-14_14-29-39_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-5 cosine, 'scalings': [100,50,50000,50000,50000,50], 
# exp_file_name = 'dgu/2024-11-14_14-30-32_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-5 cosine, 'scalings': [1,1,1,1,1,1]
# # exp_file_name = 'dgu/2024-11-14_14-48-35_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-5 cosine, 'scalings': [1,1,50,50,50,1]
# # exp_file_name = 'dgu/2024-11-14_15-02-06_phdae_dgu'  # batch=128, pen_g=1e-2, lr=1e-5 cosine, 'scalings': [1,1,50,50,50,1], 3e5 epochs
# exp_file_name = 'dgu/2024-11-14_15-05-04_phdae_dgu'  # batch=128, pen_g=1e-2, lr=1e-5 cosine, 'scalings': [1,1,10,10,10,1], 3e5 epochs
# exp_file_name = 'dgu/2024-11-14_16-30-53_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-4 cosine, 1e5 epochs
# exp_file_name = 'dgu/2024-11-14_16-36-11_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-5 cosine, 1e5 epochs
# exp_file_name = 'dgu/2024-11-14_17-32-22_phdae_dgu' # batch=64, pen_g=1e-1, lr=1e-4 cosine, 1e5 epochs
# # exp_file_name = 'dgu/2024-11-14_17-33-17_phdae_dgu' # batch=64, pen_g=1e-1, lr=1e-5 cosine, 1e5 epochs
# # exp_file_name = 'dgu/2024-11-14_15-02-06_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-4 cosine, 1e5 epochs, [100,50,50000,50000,50000,50],
# exp_file_name = 'dgu/2024-11-14_17-53-31_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-3 cosine, 1e5 epochs

############################################ TODO
# exp_file_name = 'dgu/2024-11-14_19-34-24_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-4 cosine, 1e5 epochs, new data
# exp_file_name = 'dgu/2024-11-14_19-44-24_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-5 cosine, 1e5 epochs, new data
# exp_file_name = 'dgu/2024-11-14_19-46-35_phdae_dgu' # batch=64, pen_g=1e-2, lr=1e-5 cosine, 1e5 epochs, new data
# exp_file_name = 'dgu/2024-11-14_19-53-58_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-4 cosine, 1e5 epochs, new data, scale=5 NOTE this is good
# exp_file_name = 'dgu/2024-11-14_19-55-47_phdae_dgu'  # batch=128, pen_g=1e-1, lr=1e-4 cosine, 1e5 epochs, new data
# exp_file_name = 'dgu/2024-11-14_19-57-22_phdae_dgu' # batch=128, pen_g=1e-3, lr=1e-4 cosine, 1e5 epochs, new data
# exp_file_name = 'dgu/2024-11-14_20-20-31_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-4 cosine, 1e5 epochs, new data, scale=10
# exp_file_name = 'dgu/2024-11-14_20-21-20_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-4 cosine, 3e5 epochs, new data, scale=5
# exp_file_name = 'dgu/2024-11-14_22-20-04_phdae_dgu' # batch=256, pen_g=1e-2, lr=1e-4 cosine, 3e5 epochs, new data, scale=5
# exp_file_name = 'dgu/2024-11-14_22-20-59_phdae_dgu' # batch=128, pen_g=1e-1, lr=1e-4 cosine, 3e5 epochs, new data, scale=5
# exp_file_name = 'dgu/2024-11-15_08-24-34_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-4 cosine, 3e5 epochs, new data, scale=2
# exp_file_name = 'dgu/2024-11-15_08-25-32_phdae_dgu' # batch=128, pen_g=1e-3, lr=1e-4 cosine, 3e5 epochs, new data, scale=5
# exp_file_name = 'dgu/2024-11-15_08-28-13_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-3 cosine, 3e5 epochs, new data, scale=5
# exp_file_name = 'dgu/2024-11-15_10-36-05_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-3 cosine, 3e5 epochs, new more data, scale=5
# exp_file_name = 'dgu/2024-11-15_10-47-09_phdae_dgu' # batch=256, pen_g=1e-2, lr=1e-3 cosine, 3e5 epochs, new more data, scale=5
# exp_file_name = 'dgu/2024-11-15_10-51-37_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-4 cosine, 1e5 epochs, new less data, scale=5
# exp_file_name = 'dgu/2024-11-15_10-59-07_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-4 cosine, 1e5 epochs, new less data, scale=1
# exp_file_name = 'dgu/2024-11-15_11-00-51_phdae_dgu' # batch=64, pen_g=1e-2, lr=1e-4 cosine, 1e5 epochs, new less data, scale=1
# exp_file_name = 'dgu/2024-11-15_11-10-25_phdae_dgu' # batch=64, pen_g=1e-2, lr=1e-3 cosine, 1e5 epochs, new less data, scale=1
# exp_file_name = 'dgu/2024-11-15_12-43-21_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-4 cosine, 1e5 epochs, new less data, scale=5
# exp_file_name = 'dgu/2024-11-15_12-48-48_phdae_dgu' # batch=128, pen_g=1e-2, lr=1e-4 cosine, 1e5 epochs, new less data, scale=10
# exp_file_name = 'dgu/2024-11-15_13-00-41_phdae_dgu'
# exp_file_name = 'dgu/2024-11-15_13-03-06_phdae_dgu'
# exp_file_name = 'dgu/2024-11-15_13-06-07_phdae_dgu'
# exp_file_name = 'dgu/2024-11-15_13-25-40_phdae_dgu'
# exp_file_name = 'dgu/2024-11-15_13-33-30_phdae_dgu'
# exp_file_name = 'dgu/2024-11-15_14-04-18_phdae_dgu'
# exp_file_name = 'dgu/2024-11-15_14-14-47_phdae_dgu'
# exp_file_name = 'dgu/2024-11-15_15-55-52_phdae_dgu'
# exp_file_name = 'dgu/2024-11-15_17-25-09_phdae_dgu' # all one component values
exp_file_name = 'dgu/2024-11-15_17-58-15_phdae_dgu'  # all one component values, lr=1e-3 # GOOD

exp_file_name = 'dgu_simple/2024-11-16_11-20-13_more_batch128_lr1e-3' # NOTE BASELINE

# exp_file_name = 'dgu_simple/2024-11-16_11-11-15_phdae_dgu' # lr=1e-3, batch=256, less data
# exp_file_name = 'dgu_simple/2024-11-16_11-15-12_phdae_dgu' # lr=1e-4, batch=128 # NOTE BASELINE LESS DATA SIMPLE
# exp_file_name = 'dgu_realistic/2024-11-16_11-17-43_batch128_lr1e-4'
# exp_file_name = 'dgu_realistic/2024-11-16_11-33-56_batch128_lr1e-4' # with scaling
# exp_file_name = 'dgu_realistic/2024-11-16_13-02-30_more_batch128_lr1e-5'

# exp_file_name = 'dgu_simple/2024-11-16_14-41-06_more_batch128_lr1e-3' # with slightly more realistic data
# exp_file_name = 'dgu_simple/2024-11-16_14-41-06_more_batch128_lr1e-3'
# exp_file_name = 'dgu_simple/2024-11-16_15-27-46_more_batch128_lr1e-3'
# exp_file_name = 'dgu_simple/2024-11-16_16-00-36_more_batch128_lr1e-3'
# exp_file_name = 'dgu_simple/2024-11-16_16-18-06_more_batch128_lr1e-3'
# exp_file_name = 'fhn/2024-11-15_16-01-41_fitz_hugh_nagano_less_data'
# exp_file_name = 'dgu_simple/2024-11-16_16-30-35_more_batch128_lr1e-3'
# exp_file_name = 'dgu_simple/2024-11-16_16-59-13_more_batch128_lr1e-3'
############################################
# exp_file_name = 'dgu_realistic/2024-11-16_21-51-05_more_batch128_lr1e-5' # batch=128, lr=1e-5
# exp_file_name = 'dgu_realistic/2024-11-16_21-51-59_more_batch128_lr1e-5' # batch=128, lr=1e-6
# exp_file_name = 'dgu_realistic/2024-11-16_21-54-59_random_batch128_lr1e-5' # random control, batch=128, lr=1e-5
# exp_file_name = 'dgu_realistic/2024-11-16_21-53-45_more_batch128_lr1e-5' # random control, batch=128, lr=1e-6
# exp_file_name = 'dgu_realistic/2024-11-17_09-20-27_random_batch128_lr1e-4' # batch=128, lr=1e-4, scale=5
# exp_file_name = 'dgu_realistic/2024-11-17_09-20-00_random_batch128_lr1e-5' # batch=128, lr=1e-5, scale=5
# exp_file_name = 'dgu_realistic/2024-11-17_09-21-45_batch128_lr1e-4' # batch=128, lr=1e-4
# exp_file_name = 'dgu_realistic/2024-11-17_09-22-25_random_batch128_lr1e-4' # random control, batch=128, lr=1e-5

# exp_file_name = 'fhn/2024-11-11_22-25-28_fitz_hugh_nagano' # NOTE GOOD
# exp_file_name = 'fhn/2024-11-12_11-45-05_fitz_hugh_nagano_less_data' # batch=256, pen_g=1e-1, lr=1e-5, less data
# exp_file_name = 'fhn_node/2024-11-12_21-31-38_train_node_fhn' # batch=256, pen_g=1e-1,lr=1e-3 cosine, less data
# exp_file_name = 'fhn/2024-11-14_16-53-58_fitz_hugh_nagano_less_data' # batch=128, pen_g=1e-3, lr=1e-4 cosine 
# exp_file_name = 'fhn/2024-11-14_17-29-29_fitz_hugh_nagano_less_data' # batch=128, pen_g=1e-1, lr=1e-4 cosine
# exp_file_name = 'fhn/2024-11-14_16-55-58_fitz_hugh_nagano_less_data' # batch=64, pen_g=1e-2, lr=1e-4 cosine 
# # exp_file_name = 'fhn_node/2024-11-14_16-57-33_train_node_fhn' # batch=64, lr=1e-4 cosine 
# exp_file_name = 'fhn/2024-11-14_17-29-29_fitz_hugh_nagano_less_data'
# exp_file_name = 'fhn/2024-11-14_18-27-21_fitz_hugh_nagano_less_data' # batch=128, pen_g=1e-2, lr=1e-4 cosine (same as dgu)

##### Experiment 1
exp_file_name = 'batch_training/fhn_node/0_train_node_fhn'
exp_file_name = 'fhn_node/2024-11-19_12-04-51_train_node_fhn'
exp_file_name = 'fhn_node/2024-11-19_12-59-57_train_node_fhn'

#### Experiment 2
# exp_file_name = 'dgu_simple/2024-11-17_16-43-40_baseline' # batch=128, pen_g=1e-2, lr=1e-4 cosine
exp_file_name = 'batch_training/dgu_simple/0_baseline'
exp_file_name = 'dgu_simple/2024-11-29_11-48-37_baseline'

tikz = False

def save_plot(fig, tikz: bool, name: str):
    if tikz:
        tikzplotlib_fix_ncols(fig)
        return tikzplotlib.save(name+".tex")
    else:
        return plt.savefig(name+".png", dpi=600)


sacred_save_path = os.path.abspath(os.path.join('../cyrus_experiments/runs/', exp_file_name, '1'))
metrics_file_str = os.path.abspath(os.path.join(sacred_save_path, 'metrics.json'))

with open(metrics_file_str, 'r') as f:
    results = json.load(f)

# Plot the training results
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
# for key in results.keys():
#     # if key == 'testing.loss': continue # Plot the testing loss last.
#     if 'training' in key: continue
#     if 'normalized' in key: continue
#     ax.plot(results[key]['steps'], results[key]['values'], label=key)
ax.plot(
    results['training.total_loss']['steps'][::10], 
    results['training.total_loss']['values'][::10], 
    label='training.total_loss', 
)
ax.plot(
    results['testing.total_loss']['steps'][::10],
    results['testing.total_loss']['values'][::10],
    label='testing.total_loss',
    linewidth=3,
)

ax.set_yscale('log')
ax.grid()
ax.set_title('Total loss')
ax.legend()
save_plot(fig, tikz, "training_curves_total")
plt.cla()
print('Min total loss', min(results['training.total_loss']['values']))

# Data loss
ax.plot(
    results['training.data_loss']['steps'][::10], 
    results['training.data_loss']['values'][::10], 
    label='training.data_loss', 
)
ax.plot(
    results['testing.data_loss']['steps'][::10],
    results['testing.data_loss']['values'][::10],
    label='testing.data_loss',
    linewidth=3,
)
ax.set_yscale('log')
ax.grid()
ax.legend()
ax.set_title('MSE on states')
save_plot(fig, tikz, "training_curves_data")
plt.cla()
print('Min data loss', min(results['training.data_loss']['values']))


# g loss
ax.plot(
    results['training.g_loss']['steps'][::10], 
    results['training.g_loss']['values'][::10], 
    label='training.g_loss', 
)
ax.plot(
    results['testing.g_loss']['steps'][::10],
    results['testing.g_loss']['values'][::10],
    label='testing.g_loss',
    linewidth=3,
)
ax.set_yscale('log')
ax.grid()
ax.legend()
ax.set_title('Constraint violation')
save_plot(fig, tikz, "training_curves_g")
plt.cla()
# ax.plot(
#     results['testing.data_loss']['steps'],
#     results['testing.data_loss']['values'],
#     label='testing.data_loss',
# )
print('Min g loss', min(results['training.g_loss']['values']))