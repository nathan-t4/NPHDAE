import optax
import ml_collections

def create_gnn_config(args) -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.paths = ml_collections.ConfigDict({
        'dir': args.dir,
        'training_data_path': 'results/5_mass_spring_data/train_3000_0.1_0.5_all_random_continuous.pkl',
        'evaluation_data_path': 'results/5_mass_spring_data/val_20_0.1_0.5_all_random_continuous.pkl',
    })
    config.training_params = ml_collections.ConfigDict({
        'net_name': 'GNS',
        'trial_name': '5_msd_no_globals_3000',
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
        'latent_size': 16,
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