import numpy as np
def merge_datasets(dataset1, dataset2, params=('m1', 'm2', 'k1', 'k2', 'b1', 'b2')):
    """ Only merging config and state trajectories atm """


    for k in dataset1['config'].keys():
        if k in params:
            dataset1['config'][k] = np.concatenate((dataset1['config'][k], dataset2['config'][k]), axis=0)
            
    dataset1['state_trajectories'] = np.concatenate((dataset1['state_trajectories'], dataset2['state_trajectories']), axis=0)

    dataset1['control_inputs'] = np.concatenate((dataset1['control_inputs'], dataset2['control_inputs']), axis=0)

    return dataset1