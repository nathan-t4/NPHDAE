import numpy as np
def merge_datasets(dataset1, dataset2, params=('m1', 'm2', 'k1', 'k2', 'b1', 'b2'), axis=0):
    """ Only merging config and state trajectories atm """


    for k in dataset1['config'].keys():
        if k in params:
            dataset1['config'][k] = np.concatenate((dataset1['config'][k], dataset2['config'][k]), axis=axis)
            
    dataset1['state_trajectories'] = np.concatenate((dataset1['state_trajectories'], dataset2['state_trajectories']), axis=axis)

    dataset1['control_inputs'] = np.concatenate((dataset1['control_inputs'], dataset2['control_inputs']), axis=axis)

    return dataset1

def get_train_param(rng, mean, scale, shape=()):
    N = len(mean)
    train_range = []
    for i in range(N):
        range_one = rng.uniform(mean[i] - scale, 
                                mean[i] + scale,
                                size=shape)
        train_range.append(range_one)
    return train_range

def get_validation_param(rng, mean, train_scale, val_scale, shape=()):
    N = len(mean)
    sampler = lambda a, b, shape: rng.choice(np.array([a, b]), shape)
    validation_range = []
    for i in range(N):
        range_one = rng.uniform(mean[i] + train_scale, 
                                mean[i] + train_scale + val_scale)
        range_two = rng.uniform(mean[i] - train_scale - val_scale, 
                                mean[i] - train_scale)
        validation_range.append(sampler(range_one, range_two, shape))
    return validation_range