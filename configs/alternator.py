import ml_collections
from time import strftime

def get_alternator_config(args):
    config = ml_collections.ConfigDict()
    config.seed = 0
    config.system_name = 'Alternator'
    config.n_train = 200
    config.steps = 500
    config.n_val = 20
    config.log_every_steps = 1
    config.eval_every_steps = 2
    config.ckpt_every_steps = 5
    config.clear_cache_every_steps = 1
    config.optimizer = 'adam'

    config.paths = ml_collections.ConfigDict()
    config.paths.dir = args.dir
    config.paths.ckpt_step = args.ckpt_step
    config.paths.training_data_path = f'results/{config.system_name}_data/train_{config.n_train}_{config.steps}_constant_params.pkl'
    config.paths.evaluation_data_path = f'results/{config.system_name}_data/val_{config.n_val}_1500_constant_params.pkl'

    config.training_params = ml_collections.ConfigDict()
    config.training_params.learn_matrices = False
    config.training_params.net_name = 'GNS'
    config.training_params.loss_function = 'state'
    config.training_params.num_epochs = 50
    config.training_params.min_epochs = 50
    config.training_params.batch_size = 3
    config.training_params.rollout_timesteps = 1500

    config.optimizer_params = ml_collections.ConfigDict()
    config.optimizer_params.learning_rate = 0.0001

    config.net_params = ml_collections.ConfigDict()
    config.net_params.edge_idxs = None # will set later
    config.net_params.node_idxs = None # will set later
    config.net_params.graph_from_state = None # will set later
    config.net_params.J = None # will set later
    config.net_params.R = None # will set later
    config.net_params.g = None # will set later
    config.net_params.learn_nodes = True
    config.net_params.integration_method = 'adam_bashforth'
    config.net_params.dt = 0.01
    config.net_params.T = 2
    config.net_params.num_mp_steps = 1 # TODO: fix naming for multiple mp steps
    config.net_params.noise_std = 1e-5
    config.net_params.latent_size = 8
    config.net_params.hidden_layers = 2
    config.net_params.activation = 'squareplus'
    config.net_params.use_edge_model = True
    config.net_params.use_global_model = False
    config.net_params.layer_norm = True
    config.net_params.shared_params = False
    config.net_params.dropout_rate = 0.5

    config.trial_name =  f'{strftime("%m%d-%H%M")}_T={config.net_params.T}'

    return config