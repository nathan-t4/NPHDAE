import sys
sys.path.append('..')

import cyrus_experiments.experiment_setup_files.train_phdae_dgu_realistic as dgu_realistic
import cyrus_experiments.experiment_setup_files.train_chua as chua
import cyrus_experiments.experiment_setup_files.train_fhn as fhn

training_config_factory = {
    'dgu': dgu_realistic.exp_config,
    'chua': chua.exp_config,
    'fhn': fhn.exp_config,
}

def get_config_factory(config_name):
    assert config_name in training_config_factory, f"{config_name} is not a valid training config name"
    return training_config_factory[config_name]