import ml_collections
import jax.numpy as jnp
from time import strftime

def get_rlc_config(args):
    config = ml_collections.ConfigDict()
    config.seed = 0
    config.system_name = 'RLC'
    config.n_train = 500
    config.steps = 700
    config.n_val = 20
    config.log_every_steps = 1
    config.eval_every_steps = 2
    config.ckpt_every_steps = 5
    config.clear_cache_every_steps = 1
    config.optimizer = 'adam'
    config.AC = jnp.array([[-1.0], [0.0], [0.0], [1.0]])
    config.AR = jnp.array([[0.0], [1.0], [-1.0], [0.0]])
    config.AL = jnp.array([[0.0], [0.0], [1.0], [-1.0]])
    config.AV = jnp.array([[-1.0], [1.0], [0.0], [0.0]])
    config.AI = jnp.array([[0.0], [0.0], [0.0], [0.0]])

    config.paths = ml_collections.ConfigDict()
    config.paths.dir = args.dir
    config.paths.ckpt_step = args.ckpt_step
    # config.paths.training_data_path = f'results/{config.system_name}_data/train_{config.n_train}_{config.steps}.pkl'
    # config.paths.evaluation_data_path = f'results/{config.system_name}_data/val_{config.n_val}_800.pkl'

    config.paths.training_data_path = f'results/rlc_dae_data/train_{config.n_train}_{config.steps}.pkl'
    config.paths.evaluation_data_path = f'results/rlc_dae_data/val_{config.n_val}_800.pkl'
    # config.paths.evaluation_data_path = config.paths.training_data_path

    config.training_params = ml_collections.ConfigDict()
    config.training_params.learn_matrices = False
    config.training_params.net_name = 'GNS'
    config.training_params.loss_function = 'state'
    config.training_params.num_epochs = 100
    config.training_params.min_epochs = 50
    config.training_params.batch_size = 2
    config.training_params.rollout_timesteps = 800

    config.optimizer_params = ml_collections.ConfigDict()
    config.optimizer_params.learning_rate = 0.0001

    config.net_params = ml_collections.ConfigDict()
    config.net_params.edge_idxs = None
    config.net_params.node_idxs = None
    config.net_params.include_idxs = None
    config.net_params.state_to_graph = None
    config.net_params.graph_to_state = None
    config.net_params.learn_nodes = True
    config.net_params.integration_method = 'adam_bashforth' # 'adam_bashforth'
    config.net_params.dt = 0.01
    config.net_params.T = 1
    config.net_params.num_mp_steps = 2
    config.net_params.noise_std = 1e-5
    config.net_params.latent_size = 4
    config.net_params.hidden_layers = 2
    config.net_params.activation = 'squareplus'
    config.net_params.use_edge_model = True
    config.net_params.use_global_model = False
    config.net_params.layer_norm = True
    config.net_params.shared_params = False
    config.net_params.dropout_rate = 0.5

    config.trial_name =  f'{strftime("%m%d-%H%M")}'


    return config