import os
import pickle
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

def scale_data(dataset, keys, axis, save_path):
    scaled_dataset = deepcopy(dataset)

    for key in keys:
        data = dataset[key]
        mean = np.mean(data, axis=axis)
        std = np.std(data, axis=axis)
        std = np.nan_to_num(std, nan=1)
        std[std == 0.0] = 1
        print(mean, std)
        scaled_data = (data - mean) / std
        scaled_dataset[key] = scaled_data
    
    with open(save_path, 'wb') as f:
        pickle.dump(scaled_dataset, f)

    scaled_mean = np.mean(scaled_data, axis=axis)
    scaled_std = np.std(scaled_data, axis=axis)
    return scaled_dataset
    

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    dataset = np.load(args.path, allow_pickle=True)

    path_split = os.path.split(args.path)

    save_path = os.path.join(path_split[0], f'scaled_{path_split[1]}')

    scaled_dataset = scale_data(dataset, keys=("state_trajectories",), axis=(0,1), save_path=save_path)

    original_trajs = dataset['state_trajectories']
    for i in range(30):
        plt.plot(original_trajs[i])
    plt.savefig('original_data.png')
    plt.clf()

    scaled_trajs = scaled_dataset['state_trajectories']
    for i in range(30):
        plt.plot(scaled_trajs[i])
    plt.savefig('scaled_data.png')
    plt.clf()
