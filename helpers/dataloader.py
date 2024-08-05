from cgi import test
import os
import pickle
from tkinter import E
import sacred
from tqdm import tqdm

import jax.numpy as jnp

from abc import abstractmethod

class DataLoader():
    
    def __init__(self, 
                dataset_setup : dict) -> None:
        self.dataset_setup = dataset_setup.copy()

    @abstractmethod
    def load_dataset(self) -> dict:
        """
        Load dataset.

        Returns
        -------
        dataset : 
            A dictionary containing the dataset.
        """
    
    def load_from_pickle(
            self, 
            dataset_path : str, 
            file_name : str
        ) -> dict:
        """
        Load a dataset from a pickle file. This code assumes the 
        dataset is stored as a dictionary.

        Parameters
        ----------
        dataset_path :
            The path to the pickle file containing the trajectories.
        file_name :
            The name of the pickle file containing the trajectories.

        Returns
        -------
        dataset :
            The dictionary conatining the dataset.
        """
        dataset_full_path = os.path.abspath(
                                os.path.join(
                                    dataset_path,
                                    file_name
                                )
                            )

        with open(dataset_full_path, 'rb') as f:
            dataset = pickle.load(f)

        dataset['path'] = dataset_full_path

        return dataset

class PikcleDataLoader(DataLoader):

    def __init__(self, 
                dataset_setup : dict) -> None:
        super().__init__(dataset_setup)

    def load_dataset(self) -> dict:
        """
        Load dataset.

        Returns
        -------
        dataset : 
            A dictionary containing the dataset.
        """
        dataset = self.load_from_pickle(
                        self.dataset_setup['dataset_path'],
                        self.dataset_setup['dataset_file_name']
                    )
        return dataset['train_dataset'], dataset['test_dataset']

class TrajectoryDataLoader(DataLoader):
    """
    Class for loading trajectory data.
    """
    def __init__(self, 
                dataset_setup : dict) -> None:
        super().__init__(dataset_setup)

    def load_dataset(self) -> tuple:
        """
        Load dataset. Loads the dataset specified within the dataset_setup dictionary.

        Returns
        -------
        dataset : 
            A dictionary containing the dataset.
        """
        try:
            dataset_path = self.dataset_setup['dataset_path']
        except:
            "Dataset path not specified in dataset_setup dictionary."
        try:
            train_dataset_file_name = self.dataset_setup['train_dataset_file_name']
        except:
            "Train dataset file name not specified in dataset_setup dictionary."
        try:
            test_dataset_file_name = self.dataset_setup['test_dataset_file_name']
        except:
            "Test dataset file name not specified in dataset_setup dictionary."
        train_trajectories = self.load_from_pickle(dataset_path, train_dataset_file_name)
        test_trajectories = self.load_from_pickle(dataset_path, test_dataset_file_name)

        # Specify a specific number of trajectories to use within the datasets.
        if 'num_training_trajectories' in self.dataset_setup:
            num_training_trajectories = self.dataset_setup['num_training_trajectories']
        else:
            num_training_trajectories = train_trajectories['state_trajectories'].shape[0]
        
        if 'num_testing_trajectories' in self.dataset_setup:
            num_testing_trajectories = self.dataset_setup['num_testing_trajectories']
        else:
            num_testing_trajectories = test_trajectories['state_trajectories'].shape[0]

        train_dataset = {
            'inputs' : train_trajectories['state_trajectories'][0:num_training_trajectories, :-1, :],
            'outputs' : train_trajectories['state_trajectories'][0:num_training_trajectories, 1:, :],
            'timesteps' : train_trajectories['timesteps'][0:num_training_trajectories, :-1],
            'config' : train_trajectories['config'],
        }
        if 'control_inputs' in train_trajectories:
            train_dataset['control_inputs'] = train_trajectories['control_inputs'][0:num_training_trajectories, :-1, :]

        test_dataset = {
            'inputs' : test_trajectories['state_trajectories'][0:num_testing_trajectories, :-1, :],
            'outputs' : test_trajectories['state_trajectories'][0:num_testing_trajectories, 1:, :],
            'timesteps' : train_trajectories['timesteps'][0:num_training_trajectories, :-1],
            'config' : test_trajectories['config'],
        }
        if 'control_inputs' in test_trajectories:
            test_dataset['control_inputs'] = test_trajectories['control_inputs'][0:num_testing_trajectories, :-1, :]

        train_dataset = self.reshape_dataset(train_dataset)
        print('Train dataset input shape: {}'.format(train_dataset['inputs'].shape))
        print('Train dataset output shape: {}'.format(train_dataset['outputs'].shape))
        test_dataset = self.reshape_dataset(test_dataset)
        print('Test dataset input shape: {}'.format(test_dataset['inputs'].shape))
        print('Test dataset output shape: {}'.format(test_dataset['outputs'].shape))

        return train_dataset, test_dataset

    def reshape_dataset(self, dataset : dict) -> dict:
        """
        Reshape the dataset's input and output tensors to be 2D. The first index
        represents the index of the datapoint in the dataset, the second indexes
        the dimensions of the datapoints.

        Parameters
        ----------
        dataset :
            The dataset to reshape. It should be a dictionary with dataset['inputs']
            and dataset['outputs'] arrays containing the data. The last index of
            these arrays should index the various dimensions of the datapoints.

        Returns
        -------
        dataset :
            The reshaped dataset dictionary.
        """
        in_dim = dataset['inputs'].shape[-1]
        out_dim = dataset['outputs'].shape[-1]

        dataset['inputs'] = dataset['inputs'].reshape(-1, in_dim)
        dataset['outputs'] = dataset['outputs'].reshape(-1, out_dim)

        if 'control_inputs' in dataset:
            control_dim = dataset['control_inputs'].shape[-1]
            dataset['control_inputs'] = dataset['control_inputs'].reshape(-1, control_dim)

        return dataset
    
class TrajectoryDataLoaderIncludeTimestepsInInput(DataLoader):
    """
    Class for loading trajectory data.
    """
    def __init__(self, 
                dataset_setup : dict) -> None:
        super().__init__(dataset_setup)

    def load_dataset(self) -> tuple:
        """
        Load dataset. Loads the dataset specified within the dataset_setup dictionary.

        Returns
        -------
        dataset : 
            A dictionary containing the dataset.
        """
        try:
            dataset_path = self.dataset_setup['dataset_path']
        except:
            "Dataset path not specified in dataset_setup dictionary."
        try:
            train_dataset_file_name = self.dataset_setup['train_dataset_file_name']
        except:
            "Train dataset file name not specified in dataset_setup dictionary."
        try:
            test_dataset_file_name = self.dataset_setup['test_dataset_file_name']
        except:
            "Test dataset file name not specified in dataset_setup dictionary."
        train_trajectories = self.load_from_pickle(dataset_path, train_dataset_file_name)
        test_trajectories = self.load_from_pickle(dataset_path, test_dataset_file_name)

        # Specify a specific number of trajectories to use within the datasets.
        if 'num_training_trajectories' in self.dataset_setup:
            num_training_trajectories = self.dataset_setup['num_training_trajectories']
        else:
            num_training_trajectories = train_trajectories['state_trajectories'].shape[0]
        
        if 'num_testing_trajectories' in self.dataset_setup:
            num_testing_trajectories = self.dataset_setup['num_testing_trajectories']
        else:
            num_testing_trajectories = test_trajectories['state_trajectories'].shape[0]

        train_dataset = {
            'inputs' : train_trajectories['state_trajectories'][0:num_training_trajectories, :-1, :],
            'outputs' : train_trajectories['state_trajectories'][0:num_training_trajectories, 1:, :],
            'timesteps' : train_trajectories['timesteps'][0:num_training_trajectories, :-1],
            'config' : train_trajectories['config'],
        }
        if 'control_inputs' in train_trajectories:
            train_dataset['control_inputs'] = train_trajectories['control_inputs'][0:num_training_trajectories, :-1, :]

        test_dataset = {
            'inputs' : test_trajectories['state_trajectories'][0:num_testing_trajectories, :-1, :],
            'outputs' : test_trajectories['state_trajectories'][0:num_testing_trajectories, 1:, :],
            'timesteps' : test_trajectories['timesteps'][0:num_testing_trajectories, :-1],
            'config' : test_trajectories['config'],
        }
        if 'control_inputs' in test_trajectories:
            test_dataset['control_inputs'] = test_trajectories['control_inputs'][0:num_testing_trajectories, :-1, :]

        train_dataset = self.reshape_dataset(train_dataset)
        print('Train dataset input shape: {}'.format(train_dataset['inputs'].shape))
        print('Train dataset output shape: {}'.format(train_dataset['outputs'].shape))
        test_dataset = self.reshape_dataset(test_dataset)
        print('Test dataset input shape: {}'.format(test_dataset['inputs'].shape))
        print('Test dataset output shape: {}'.format(test_dataset['outputs'].shape))

        return train_dataset, test_dataset

    def reshape_dataset(self, dataset : dict) -> dict:
        """
        Reshape the dataset's input and output tensors to be 2D. The first index
        represents the index of the datapoint in the dataset, the second indexes
        the dimensions of the datapoints.

        Parameters
        ----------
        dataset :
            The dataset to reshape. It should be a dictionary with dataset['inputs']
            and dataset['outputs'] arrays containing the data. The last index of
            these arrays should index the various dimensions of the datapoints.

        Returns
        -------
        dataset :
            The reshaped dataset dictionary.
        """
        dataset['inputs'] = dataset['inputs'].reshape(-1, dataset['inputs'].shape[-1])
        dataset['inputs'] = jnp.concatenate((dataset['inputs'], dataset['timesteps'].reshape(-1, 1)), axis=1)

        dataset['outputs'] = dataset['outputs'].reshape(-1, dataset['outputs'].shape[-1])

        return dataset


dataloader_factory = {
    'trajectory': TrajectoryDataLoader,
    'pickle_dataset' : PikcleDataLoader,
    'trajectory_timesteps_in_input' : TrajectoryDataLoaderIncludeTimestepsInInput,
    # 'supervised_regression': SupervisedRegressionDataLoader,
}

def load_dataset_from_setup(dataset_setup : dict) -> DataLoader:
    """
    Load the dataset given the configuration options
    specified in dataset_setup dictionary.
    """
    dataloader = dataloader_factory[dataset_setup['dataset_type']](dataset_setup)
    return dataloader.load_dataset()