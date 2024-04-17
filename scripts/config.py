import optax
import ml_collections

def create_gnn_config(args) -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.paths = ml_collections.ConfigDict({
        'dir': args.dir,
        'training_data_path': 'results/2_mass_spring_data/train_3000_0.1_0.5_passive.pkl',
        'evaluation_data_path': 'results/2_mass_spring_data/val_20_0.1_0.5_passive.pkl',
    })
    config.training_params = ml_collections.ConfigDict({
        'net_name': 'GNS',
        'trial_name': '2_msd_no_globals_3000_passive',
        'loss_function': 'acceleration',
        'num_epochs': int(5e2),
        'min_epochs': int(30),
        'batch_size': 2,
        'rollout_timesteps': 1500,
        'log_every_steps': 1,
        'eval_every_steps': 5,
        'checkpoint_every_steps': 5,
        'clear_cache_every_steps': 1,
        'add_undirected_edges': True,
        'add_self_loops': True,
        'train_multi_trajectories': True,
    })
    config.optimizer_params = ml_collections.ConfigDict({
        'learning_rate': optax.exponential_decay(init_value=1e-3, transition_steps=5e2, decay_rate=0.1, end_value=1e-5),
    })
    config.net_params = ml_collections.ConfigDict({
        'prediction': 'acceleration',
        'integration_method': 'SemiImplicitEuler', 
        # 'horizon': 5, # for gnode only
        'vel_history': 5,
        'control_history': 5,
        'num_mp_steps': 1, # too big causes oversmoothing
        'noise_std': 0.0003,
        'latent_size': 16, # use 32 for quintuple mass spring damper, other <5 use 16
        'hidden_layers': 2,
        'activation': 'relu',
        'use_edge_model': True,
        'layer_norm': True,
        'shared_params': False,
        'dropout_rate': 0.5,
        'add_undirected_edges': config.training_params.add_undirected_edges,
        'add_self_loops': config.training_params.add_self_loops,
    })
    return config

def create_comp_gnn_config():
    config = ml_collections.ConfigDict()

    config.optimizer_params = ml_collections.ConfigDict({
        'learning_rate': optax.exponential_decay(init_value=1e-3, transition_steps=5e2, decay_rate=0.1, end_value=1e-5),
    })

    config.paths = ml_collections.ConfigDict({
        'dir_one': 'results/test_models/0412_test_gnn/2_msd_no_globals_173107',
        'evaluation_data_path_one': 'results/2_mass_spring_data/val_20_0.1_0.5_all_random_continuous.pkl',
        'dir_two': 'results/test_models/0415_test_gnn/3_msd_no_globals_3000_state_190703',
        'evaluation_data_path_two': 'results/3_mass_spring_data/val_20_0.1_0.5_all_random_continuous.pkl',
        'evaluation_data_path_comp': 'results/5_mass_spring_data/val_20_0.1_0.5_all_random_continuous.pkl',
    })

    config.eval_params = ml_collections.ConfigDict({
        'trial_name': '2+3',
        'rollout_timesteps': 1500,
    })

    config.net_params_one = ml_collections.ConfigDict({
        'prediction': 'acceleration',
        'integration_method': 'SemiImplicitEuler', 
        # 'horizon': 5, # for gnode only
        'vel_history': 5,
        'control_history': 5,
        'num_mp_steps': 1, # too big causes oversmoothing
        'noise_std': 0.0003,
        'latent_size': 16, # use 32 for quintuple mass spring damper, other <5 use 16
        'hidden_layers': 2,
        'activation': 'relu',
        'use_edge_model': True,
        'layer_norm': True,
        'shared_params': False,
        'dropout_rate': 0.5,
        'add_undirected_edges': True,
        'add_self_loops': True,
    })

    config.net_params_two = ml_collections.ConfigDict({
        'prediction': 'acceleration',
        'integration_method': 'SemiImplicitEuler', 
        # 'horizon': 5, # for gnode only
        'vel_history': 5,
        'control_history': 5,
        'num_mp_steps': 1, # too big causes oversmoothing
        'noise_std': 0.0003,
        'latent_size': 16, # use 32 for quintuple mass spring damper, other <5 use 16
        'hidden_layers': 2,
        'activation': 'relu',
        'use_edge_model': True,
        'layer_norm': True,
        'shared_params': False,
        'dropout_rate': 0.5,
        'add_undirected_edges': True,
        'add_self_loops': True,
    })
    
    return config