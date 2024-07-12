import ml_collections
from time import strftime

def get_optuna_cfg(name, trial) -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.seed = 0
    config.system_name = name
    config.n_train = 200
    config.steps = 700
    config.n_val = 20
    config.log_every_steps = 1
    config.eval_every_steps = 2
    config.ckpt_every_steps = 5
    config.clear_cache_every_steps = 1
    config.trial_name = f'{strftime("%m%d-%H%M")}_constant_optuna_{config.n_train}'

    config.paths = ml_collections.ConfigDict()
    config.paths.dir = None
    config.paths.ckpt_step = None
    config.paths.training_data_path = f'results/{config.system_name}_data/train_{config.n_train}_{config.steps}_constant_params.pkl'
    config.paths.evaluation_data_path = f'results/{config.system_name}_data/val_{config.n_val}_1500_constant_params.pkl'

    config.training_params = ml_collections.ConfigDict()
    config.training_params.net_name = 'GNS'
    config.training_params.learn_matrices = False
    config.training_params.trial_name = f'{config.n_train}_squareplus'
    config.training_params.loss_function = 'state'
    config.training_params.num_epochs = 30
    config.training_params.min_epochs = 30
    config.training_params.batch_size = trial.suggest_int('batch_size', 1, 100, log=True)
    config.training_params.rollout_timesteps = 1500 

    config.optimizer_params = ml_collections.ConfigDict()
    config.optimizer_params.learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    config.net_params = ml_collections.ConfigDict()
    config.net_params.dt = 0.01
    config.net_params.T = trial.suggest_int('T', 1, 5)
    config.net_params.J = None
    config.net_params.R = None
    config.net_params.g = None
    config.net_params.edge_idxs = None
    config.net_params.node_idxs = None
    config.net_params.include_idxs = None
    config.net_params.integration_method = 'adam_bashforth'
    # config.net_params.num_mp_steps = trial.suggest_int('num_mp_steps', 1, 2, log=True)
    config.net_params.num_mp_steps = 1
    config.net_params.noise_std = trial.suggest_float('noise_std', 1e-5, 1e-3, log=True)
    config.net_params.latent_size = trial.suggest_int('latent_size', 4, 8, log=True)
    config.net_params.hidden_layers = 2
    config.net_params.activation = trial.suggest_categorical('activation', ["relu", "swish", "squareplus"])
    config.net_params.learn_nodes = True
    config.net_params.use_edge_model = True
    config.net_params.use_global_model = False
    config.net_params.layer_norm = True
    config.net_params.shared_params = False
    config.net_params.dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.8)

    return config