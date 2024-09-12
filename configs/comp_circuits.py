import ml_collections
import jax.numpy as jnp
from time import strftime

def get_comp_gnn_config(args):
    config = ml_collections.ConfigDict()
    config.seed = 0

    # Define composition network
    config.subsystem_names = ['DGU', 'TL', 'DGU']
    config.learned_subsystem = [True, False, True]
    config.last_subsystem_idx = 1
    config.Alambda =jnp.array([[0.0, 0.0], 
                                [0.0, 0.0],
                                [1.0, 0.0],
                                [-1.0, 0.0],
                                [0.0, 0.0],
                                [0.0, -1.0], # changed order
                                [0.0, 0.0],
                                [0.0, 0.0],
                                [0.0, 1.0]]) # changed order

    config.trial_name = f'{strftime("%m%d-%H%M")}_{"_".join(config.subsystem_names)}'
    config.rollout_timesteps = 100
    config.log_every_steps = 1
    config.eval_every_steps = 2
    config.ckpt_every_steps = 5
    config.clear_cache_every_steps = 1

    config.paths = ml_collections.ConfigDict()
    config.paths.dir = args.dir
    config.paths.comp_data_path = 'results/microgrid_dae_data/val_5_800.pkl'
    config.paths.ckpt_steps = [16, None, 16]
    config.paths.ckpt_dirs = [
        'results/GNS/DGU/0824-1038/checkpoint/best_model',
        None,
        'results/GNS/DGU/0824-1038/checkpoint/best_model',
    ]
    config.paths.training_data_paths = [
        'dgu_dae_data/train_500_700.pkl',
        'results/DGU_TL_data/test.pkl', # empty dataset to initialize graph builder
        'dgu_dae_data/train_500_700.pkl',
    ]

    config.incidence_matrices_dgu = ml_collections.ConfigDict()
    config.incidence_matrices_dgu.AC = jnp.array([[0.0], [0.0], [1.0]])
    config.incidence_matrices_dgu.AR = jnp.array([[-1.0], [1.0], [0.0]])
    config.incidence_matrices_dgu.AL = jnp.array([[0.0], [1.0], [-1.0]])
    config.incidence_matrices_dgu.AV = jnp.array([[1.0], [0.0], [0.0]])
    config.incidence_matrices_dgu.AI = jnp.array([[0.0], [0.0], [-1.0]])

    config.optimizer_params_1 = ml_collections.ConfigDict()
    config.optimizer_params_1.learning_rate = 1e-4
    
    config.net_params_1 = ml_collections.ConfigDict()
    config.net_params_1.state_to_graph = None
    config.net_params_1.graph_to_state = None
    config.net_params_1.edge_idxs = None
    config.net_params_1.node_idxs = None
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

    config.incidence_matrices_dgu = ml_collections.ConfigDict()
    config.incidence_matrices_dgu.AC = jnp.array([[0.0], [0.0], [1.0]])
    config.incidence_matrices_dgu.AR = jnp.array([[-1.0], [1.0], [0.0]])
    config.incidence_matrices_dgu.AL = jnp.array([[0.0], [1.0], [-1.0]])
    config.incidence_matrices_dgu.AV = jnp.array([[1.0], [0.0], [0.0]])
    config.incidence_matrices_dgu.AI = jnp.array([[0.0], [0.0], [-1.0]])

    config.optimizer_params_2 = ml_collections.ConfigDict()
    config.optimizer_params_2.learning_rate = 1e-4
    
    config.net_params_2 = ml_collections.ConfigDict()
    config.net_params_2.state_to_graph = None
    config.net_params_2.graph_to_state = None
    config.net_params_2.edge_idxs = None
    config.net_params_2.node_idxs = None
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

    config.incidence_matrices_tl = ml_collections.ConfigDict()
    config.incidence_matrices_tl.AC = jnp.array([[0.0], [0.0], [0.0]])
    config.incidence_matrices_tl.AR = jnp.array([[-1.0], [1.0], [0.0]])
    config.incidence_matrices_tl.AL = jnp.array([[0.0], [-1.0], [1.0]])
    config.incidence_matrices_tl.AV = jnp.array([[0.0], [0.0], [0.0]])
    config.incidence_matrices_tl.AI = jnp.array([[0.0], [0.0], [0.0]])

    config.incidence_matrices = [
        config.incidence_matrices_dgu, 
        config.incidence_matrices_tl, 
        config.incidence_matrices_dgu
    ]
    config.net_params = [
        config.net_params_1, None, config.net_params_2
    ]
    config.optimizer_params = [
        config.optimizer_params_1, None, config.optimizer_params_2
    ]

    # TODO: check that these matrices are consistent with subsystems
    config.composite_incidence_matrices = ml_collections.ConfigDict()
    config.composite_incidence_matrices.AC = jnp.array(
                    [[0.0, 0.0], 
                    [0.0, 0.0], 
                    [1.0, 0.0], 
                    [0.0, 0.0],         
                    [0.0, 0.0], 
                    [0.0, 0.0], 
                    [0.0, 1.0]])
    config.composite_incidence_matrices.AR = jnp.array(
                    [[-1.0, 0.0, 0.0], 
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0]])
    config.composite_incidence_matrices.AL = jnp.array(
                    [[0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, -1.0]])
    config.composite_incidence_matrices.AV = jnp.array(
                    [[1.0, 0.0], 
                    [0.0, 0.0], 
                    [0.0, 0.0], 
                    [0.0, 0.0],         
                    [0.0, 1.0], 
                    [0.0, 0.0], 
                    [0.0, 0.0]])
    config.composite_incidence_matrices.AI = jnp.array(
                    [[0.0, 0.0], 
                    [0.0, 0.0], 
                    [-1.0, 0.0], 
                    [0.0, 0.0],         
                    [0.0, 0.0], 
                    [0.0, 0.0], 
                    [0.0, -1.0]])
    return config