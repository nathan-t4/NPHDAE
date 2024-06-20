import ml_collections
import jax.numpy as jnp
from time import strftime

def get_comp_gnn_config(args):
    config = ml_collections.ConfigDict()
    config.seed = 0
    config.system_name = 'LC1'
    config.eval_system_name = 'CoupledLC'
    config.set_nodes = True
    config.rollout_timesteps = 1500
    config.optimizer = 'adam'
    config.trial_name =  f'{strftime("%m%d-%H%M")}_{config.system_name}'

    config.paths = ml_collections.ConfigDict()
    config.paths.dir = args.dir
    # config.paths.ckpt_one_step = 6
    # config.paths.ckpt_one_dir = 'results/GNS/CoupledLC/0619-1032_squareplus/checkpoint/best_model'
    # config.paths.evaluation_data_path = f'results/LC1_data/val_5_1500.pkl'
    config.paths.ckpt_one_step = 12
    config.paths.ckpt_one_dir = 'results/GNS/LC1/0619-1228_optuna_200/checkpoint/best_model'
    config.paths.evaluation_data_path = 'results/CoupledLC_data/val_5_1500.pkl'

    config.training_params_1 = ml_collections.ConfigDict()
    config.training_params_1.net_name = 'GNS'
    config.training_params_1.loss_function = 'state'
    config.training_params_1.num_epochs = 500
    config.training_params_1.min_epochs = 30
    config.training_params_1.batch_size = 2
    config.training_params_1.rollout_timesteps = 1500

    config.optimizer_params_1 = ml_collections.ConfigDict()
    config.optimizer_params_1.learning_rate = 0.0005

    config.net_params_1 = ml_collections.ConfigDict()
    config.net_params_1.graph_from_state = None # will set later
    config.net_params_1.J = None # will set later
    config.net_params_1.g = None # will set later
    config.net_params_1.integration_method = 'euler'
    config.net_params_1.num_mp_steps = 1
    config.net_params_1.noise_std = 0.00033
    config.net_params_1.latent_size = 5
    config.net_params_1.hidden_layers = 2
    config.net_params_1.activation = 'swish'
    config.net_params_1.learn_nodes = False
    config.net_params_1.use_edge_model = True
    config.net_params_1.use_global_model = False
    config.net_params_1.layer_norm = True
    config.net_params_1.shared_params = False
    config.net_params_1.dropout_rate = 0.5

#     config.training_params_2 = ml_collections.ConfigDict({
#             'net_name': 'GNS',
#             'loss_function': 'state',
#             'num_epochs': int(5e2),
#             'min_epochs': int(30),
#             'batch_size': 9,
#     })
#     config.optimizer_params_2 = ml_collections.ConfigDict({
#             'learning_rate': 0.0061530848896117285,
#     })
#     config.net_params_2 = ml_collections.ConfigDict({
#             'integration_method': 'euler', 
#             'num_mp_steps': 5,
#             'noise_std': 0.0003,
#             'latent_size': 12,
#             'hidden_layers': 2,
#             'activation': 'swish',
#             'use_edge_model': True,
#             'use_global_model': False,
#             'layer_norm': True,
#             'shared_params': False,
#             'dropout_rate': 0.5,
#     })
    return config