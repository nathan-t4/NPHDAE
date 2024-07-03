import ml_collections
import jax.numpy as jnp
from time import strftime

def get_reuse_model_config(args):
    config = ml_collections.ConfigDict()
    config.seed = 0
    config.system_name = 'LC1'
    config.eval_system_name = 'CoupledLC'
    config.set_nodes = True # False
    config.rollout_timesteps = 1500
    config.optimizer = 'adam'
    config.trial_name =  f'{strftime("%m%d-%H%M")}_{config.system_name}_to_{config.eval_system_name}'

    config.paths = ml_collections.ConfigDict()
    config.paths.dir = args.dir
    # config.paths.ckpt_one_step = 6
    # config.paths.ckpt_one_dir = 'results/GNS/CoupledLC/0619-1032_squareplus/checkpoint/best_model'
    # config.paths.evaluation_data_path = f'results/LC1_data/val_5_1500.pkl'
    # config.paths.ckpt_one_step = 26
    # config.paths.ckpt_one_dir = 'results/GNS/LC1/0620-1222_200_constant/checkpoint/best_model'
    config.paths.ckpt_one_step = 30
    config.paths.ckpt_one_dir = 'results/GNS/LC1/0621-1001_constant_optuna_200/checkpoint/best_model'
    config.paths.evaluation_data_path = 'results/CoupledLC_data/val_5_1500_constant_params.pkl'

    config.training_params_1 = ml_collections.ConfigDict()
    config.training_params_1.net_name = 'GNS'
    config.training_params_1.loss_function = 'state'
    config.training_params_1.learn_matrices = False
    config.training_params_1.num_epochs = 500
    config.training_params_1.min_epochs = 30
    config.training_params_1.batch_size = 1
    config.training_params_1.rollout_timesteps = 1500

    config.optimizer_params_1 = ml_collections.ConfigDict()
    config.optimizer_params_1.learning_rate = 0.0002

    config.net_params_1 = ml_collections.ConfigDict()
    config.net_params_1.graph_from_state = None # will set later
    config.net_params_1.J = None # will set later
    config.net_params_1.g = None # will set later
    config.net_params_1.edge_idxs = None
    config.net_params_1.include_idxs = None
    config.net_params_1.integration_method = 'adam_bashforth'
    config.net_params_1.dt = 0.01
    config.net_params_1.T = 5
    config.net_params_1.num_mp_steps = 1
    config.net_params_1.noise_std = 3e-4
    config.net_params_1.latent_size = 4
    config.net_params_1.hidden_layers = 2
    config.net_params_1.activation = 'squareplus'
    config.net_params_1.learn_nodes = True
    config.net_params_1.use_edge_model = True
    config.net_params_1.use_global_model = False
    config.net_params_1.layer_norm = True
    config.net_params_1.shared_params = False
    config.net_params_1.dropout_rate = 0.5
    return config