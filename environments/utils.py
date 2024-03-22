import numpy as np
def merge_datasets(dataset1, dataset2):
    """ Only merging state trajectories atm """
    # for params
    # if dataset1['config']['m1'] is float: 
    #     dataset1['config']['m1'] = [dataset1['config']['m1']]
    
    # dataset1['config']['m1'] = np.concatenate((dataset1['config']['m1'], [dataset2['config']['m1']]))
    
    dataset1['state_trajectories'] = np.concatenate((dataset1['state_trajectories'], dataset2['state_trajectories']))
    

    return dataset1
