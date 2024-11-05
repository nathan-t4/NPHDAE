import jax

import os, sys

import pickle
from datetime import datetime
import jax.numpy as jnp
import numpy as np

sys.path.append('../')
from sacred import Experiment
from sacred.observers import FileStorageObserver
import datetime

sys.path.append('.')
from helpers.training_config_factory import get_config_factory
# from cyrus_experiments.experiment_setup_files.train_phdae_dgu import exp_config
# from cyrus_experiments.experiment_setup_files.train_phdae_dgu_realistic import exp_config
# from cyrus_experiments.experiment_setup_files.train_chua import exp_config
jax.config.update('jax_platform_name', 'gpu')

exp_to_train = input("Enter which training task (dgu or chua or fhn): ")

exp_config = get_config_factory(exp_to_train)
exp_name = exp_config['exp_name']
now = datetime.datetime.now()
datetime_exp_name = now.strftime(
    "%Y-%m-%d_%H-%M-%S_" + exp_name
)

ex = Experiment(datetime_exp_name)
ex.add_config(exp_config)

ex.observers.append(FileStorageObserver.create(f'runs/{exp_to_train}/' + datetime_exp_name))
    
@ex.automain
def main(
        _run, 
        _log
    ):

    print("Starting sacred experiment: {}".format(datetime_exp_name))

    rng_key = jax.random.PRNGKey(exp_config['exp_setup']['seed'])

    from helpers.dataloader import load_dataset_from_setup
    train_dataset, test_dataset = load_dataset_from_setup(exp_config['dataset_setup'])

    # Initialize the model to be trained.
    from helpers.model_factory import get_model_factory
    rng_key, subkey = jax.random.split(rng_key)
    model_factory = get_model_factory(exp_config['model_setup'])
    model =  model_factory.create_model(subkey)
    model.training = True

    # Create a model trainer object, which handles all of the model optimization.
    from helpers.trainer_factories import get_trainer_factory
    trainer_factory = get_trainer_factory(exp_config['trainer_setup'])
    trainer = trainer_factory.create_trainer(model)

    # Run the training algorithm
    save_path = os.path.join(os.curdir, 'runs/' + datetime_exp_name + '/1/' + 'model.pkl')
    rng_key, subkey = jax.random.split(rng_key)
    trainer.train(train_dataset,
                test_dataset,
                subkey,
                save_path=save_path,
                sacred_runner=_run)

    if not os.path.exists('temp_data'):
        os.makedirs('temp_data')
    with open('temp_data/model_params.pkl', 'wb') as f:
        pickle.dump(trainer.params, f)

    ex.add_artifact('temp_data/model_params.pkl')

    # # Save the results of the experiment
    # model_save_path = os.path.abspath('../experiments/saved_models')
    # model_file_name = datetime_experiment_name.replace(' ', '_') + '.pkl'
    # model_save_file_str = os.path.join(os.path.abspath(model_save_path), 
    #                                                     model_file_name)

    # with open(model_save_file_str, 'wb') as f:
    #     pickle.dump(trainer.params, f)

    # # Associate the outputs with the Sacred experiment
    # ex.add_artifact(model_save_file_str)