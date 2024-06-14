import ml_collections
from time import strftime

def get_lc_config(args):
        config = ml_collections.ConfigDict()
        config.seed = 0
        config.system_name = 'LC1'
        config.n_train = 200
        config.steps = 700
        config.n_val = 20
        config.log_every_steps = 1
        config.eval_every_steps = 2
        config.ckpt_every_steps = 5
        config.clear_cache_every_steps = 1
        config.optimizer = 'adam'
        config.trial_name =  f'{strftime("%m%d-%H%M")}_{config.n_train}'

        config.paths = ml_collections.ConfigDict()
        config.paths.dir = args.dir
        config.paths.ckpt_step = args.ckpt_step
        config.paths.training_data_path = f'results/{config.system_name}_data/train_{config.n_train}_{config.steps}.pkl'
        config.paths.evaluation_data_path = f'results/{config.system_name}_data/val_{config.n_val}_1500.pkl'

        config.training_params = ml_collections.ConfigDict()
        config.training_params.net_name = 'GNS'
        config.training_params.trial_name = f'{config.n_train}'
        config.training_params.loss_function = 'state'
        config.training_params.num_epochs = 500
        config.training_params.min_epochs = 30
        config.training_params.batch_size = 5
        config.training_params.rollout_timesteps = 1500

        config.optimizer_params = ml_collections.ConfigDict()
        config.optimizer_params.learning_rate = 1e-3

        config.net_params = ml_collections.ConfigDict()
        config.net_params.integration_method = 'euler'
        config.net_params.num_mp_steps = 2
        config.net_params.noise_std = 0.0003
        config.net_params.latent_size = 5
        config.net_params.hidden_layers = 2
        config.net_params.activation = 'relu'
        config.net_params.use_edge_model = True
        config.net_params.use_global_model = False
        config.net_params.layer_norm = True
        config.net_params.shared_params = False
        config.net_params.dropout_rate = 0.5

        return config