import ml_collections
from time import strftime

def get_comp_gnn_config(args):
    config = ml_collections.ConfigDict()
    config.seed = 0
    config.system_name = 'comp'
    config.trial_name = f'{strftime("%m%d-%H%M")}_{config.system_name}'
    config.rollout_timesteps = 1500
    config.log_every_steps = 1
    config.eval_every_steps = 2
    config.ckpt_every_steps = 5
    config.clear_cache_every_steps = 1

    config.paths = ml_collections.ConfigDict()
    config.paths.dir = args.dir
    config.paths.ckpt_one_step = 22
    config.paths.ckpt_one_dir = 'results/GNS/LC1/0614-0923_200/checkpoint/best_model'
    config.paths.ckpt_two_step = 6
    config.paths.ckpt_two_dir = 'results/GNS/LC2/0614-1117_400/checkpoint/best_model'
    config.paths.coupled_lc_data_path = 'results/CoupledLC_data/val_2_1500.pkl'

    config.training_params_1 = ml_collections.ConfigDict({
            'net_name': 'GNS',
            'loss_function': 'state',
            'num_epochs': int(5e2),
            'min_epochs': int(30),
            'batch_size': 5,
    })
    config.optimizer_params_1 = ml_collections.ConfigDict({
            'learning_rate': 1e-3,
    })
    config.net_params_1 = ml_collections.ConfigDict({
            'integration_method': 'euler', 
            'num_mp_steps': 2,
            'noise_std': 0.0003,
            'latent_size': 5,
            'hidden_layers': 2,
            'activation': 'relu',
            'use_edge_model': True,
            'use_global_model': False,
            'layer_norm': True,
            'shared_params': False,
            'dropout_rate': 0.5,
    })
    config.training_params_2 = ml_collections.ConfigDict({
            'net_name': 'GNS',
            'loss_function': 'state',
            'num_epochs': int(5e2),
            'min_epochs': int(30),
            'batch_size': 9,
    })
    config.optimizer_params_2 = ml_collections.ConfigDict({
            'learning_rate': 0.0061530848896117285,
    })
    config.net_params_2 = ml_collections.ConfigDict({
            'integration_method': 'euler', 
            'num_mp_steps': 5,
            'noise_std': 0.0003,
            'latent_size': 12,
            'hidden_layers': 2,
            'activation': 'swish',
            'use_edge_model': True,
            'use_global_model': False,
            'layer_norm': True,
            'shared_params': False,
            'dropout_rate': 0.5,
    })
    return config