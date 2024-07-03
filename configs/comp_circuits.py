import ml_collections
import jax.numpy as jnp
from time import strftime

def get_comp_gnn_config(args):
    config = ml_collections.ConfigDict()
    config.seed = 0
    config.net_one_name = 'LC1'
    config.net_two_name = 'LC1'
    config.learn_matrices_one = False
    config.learn_matrices_two = False
    config.trial_name = f'{strftime("%m%d-%H%M")}_comp_{config.net_one_name}_w_{config.net_two_name}'
    config.rollout_timesteps = 1500
    config.log_every_steps = 1
    config.eval_every_steps = 2
    config.ckpt_every_steps = 5
    config.clear_cache_every_steps = 1

    config.paths = ml_collections.ConfigDict()
    config.paths.dir = args.dir
    config.paths.ckpt_one_step = 38
    config.paths.ckpt_one_dir = 'results/GNS/LC1/0627_T=1_learn_nodes/checkpoint/best_model'
    config.paths.ckpt_two_step = 38
    config.paths.ckpt_two_dir = 'results/GNS/LC1/0627_T=1_learn_nodes/checkpoint/best_model'
    config.paths.coupled_lc_data_path = 'results/CoupledLC_data/val_5_1500_constant_params.pkl'

    config.optimizer_params_1 = ml_collections.ConfigDict()
    config.optimizer_params_1.learning_rate = 1e-4
    
    config.net_params_1 = ml_collections.ConfigDict()
    config.net_params_1.graph_from_state = None
    config.net_params_1.J = None
    config.net_params_1.R = None
    config.net_params_1.g = None
    config.net_params_1.edge_idxs = None
    config.net_params_1.include_idxs = None
    config.net_params_1.integration_method = 'adam_bashforth'
    config.net_params_1.dt = 0.01
    config.net_params_1.T = 1
    config.net_params_1.num_mp_steps = 1
    config.net_params_1.noise_std = 1e-5
    config.net_params_1.latent_size = 4
    config.net_params_1.hidden_layers = 2
    config.net_params_1.activation = 'squareplus'
    config.net_params_1.learn_nodes = True
    config.net_params_1.use_edge_model = True
    config.net_params_1.use_global_model = False
    config.net_params_1.layer_norm = True
    config.net_params_1.shared_params = False
    config.net_params_1.dropout_rate = 0.5

    config.optimizer_params_2 = ml_collections.ConfigDict()
    config.optimizer_params_2.learning_rate = 1e-4
    
    config.net_params_2 = ml_collections.ConfigDict()
    config.net_params_2.graph_from_state = None
    config.net_params_2.J = None
    config.net_params_2.R = None
    config.net_params_2.g = None
    config.net_params_2.edge_idxs = None
    config.net_params_2.include_idxs = None
    config.net_params_2.integration_method = 'adam_bashforth'
    config.net_params_2.dt = 0.01
    config.net_params_2.T = 1
    config.net_params_2.num_mp_steps = 1
    config.net_params_2.noise_std = 1e-5
    config.net_params_2.latent_size = 4
    config.net_params_2.hidden_layers = 2
    config.net_params_2.activation = 'squareplus'
    config.net_params_2.learn_nodes = True
    config.net_params_2.use_edge_model = True
    config.net_params_2.use_global_model = False
    config.net_params_2.layer_norm = True
    config.net_params_2.shared_params = False
    config.net_params_2.dropout_rate = 0.5
    return config