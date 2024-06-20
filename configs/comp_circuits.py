import ml_collections
import jax.numpy as jnp
from time import strftime

def get_comp_gnn_config(args):
    config = ml_collections.ConfigDict()
    config.seed = 0
    config.net_one_name = 'LC1'
    config.net_two_name = 'LC1'
    config.trial_name = f'{strftime("%m%d-%H%M")}_comp_{config.net_one_name}_w_{config.net_two_name}'
    config.rollout_timesteps = 1500
    config.log_every_steps = 1
    config.eval_every_steps = 2
    config.ckpt_every_steps = 5
    config.clear_cache_every_steps = 1

    config.paths = ml_collections.ConfigDict()
    config.paths.dir = args.dir
    config.paths.ckpt_one_step = 14
    config.paths.ckpt_one_dir = 'results/GNS/LC1/0620-1100_200_constant/checkpoint/best_model'
    config.paths.ckpt_two_step = 14
    config.paths.ckpt_two_dir = 'results/GNS/LC1/0620-1100_200_constant/checkpoint/best_model'
    config.paths.coupled_lc_data_path = 'results/CoupledLC_data/val_5_1500_constant_params.pkl'

    config.training_params_1 = ml_collections.ConfigDict({
            'net_name': 'GNS',
            'loss_function': 'state',
            'num_epochs': int(5e2),
            'min_epochs': int(30),
            'batch_size': 32, 
    })
    config.optimizer_params_1 = ml_collections.ConfigDict({
            'learning_rate': 7e-3,
    })
    config.net_params_1 = ml_collections.ConfigDict({
            'graph_from_state': None,
            'J': jnp.array([[0, 1, 0],
                            [-1, 0, 1],
                            [0, -1, 0]]),
            'g': jnp.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]]), # TODO: changed last entry from -1 to 0
            'edge_idxs': None,
            'include_idxs': None,
            'integration_method': 'euler', 
            'dt': 0.01,
            'T': 1,
            'num_mp_steps': 1,
            'noise_std': 1e-5,
            'latent_size': 4,
            'hidden_layers': 2,
            'activation': 'squareplus',
            'learn_nodes': False,
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
            'batch_size': 32,
    })
    config.optimizer_params_2 = ml_collections.ConfigDict({
            'learning_rate': 7e-3,
    })
    config.net_params_2 = ml_collections.ConfigDict({
            'graph_from_state': None,
            'J': jnp.array([[0, 1, 0],
                            [-1, 0, 1],
                            [0, -1, 0]]),
            'g': jnp.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]]), # TODO changed last entry from -1 to 0
            'edge_idxs': None,
            'include_idxs': None,
            'integration_method': 'euler', 
            'dt': 0.01,
            'T': 1,
            'num_mp_steps': 1,
            'noise_std': 0.0009,
            'latent_size': 4,
            'hidden_layers': 2,
            'activation': 'squareplus',
            'learn_nodes': False,
            'use_edge_model': True,
            'use_global_model': False,
            'layer_norm': True,
            'shared_params': False,
            'dropout_rate': 0.5,
    })
    return config