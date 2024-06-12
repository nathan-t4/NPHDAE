import optax
import ml_collections

def create_gnn_config(args) -> ml_collections.ConfigDict:
    if 'lc' in args.system.lower():
        config = ml_collections.ConfigDict({
            'system_name': 'CoupledLC',
            'n_train': 200,
            'steps': 700,
            'n_val': 20,
            'ckpt_step': args.ckpt_step,
        })
        config.paths = ml_collections.ConfigDict({
            'dir': args.dir,
            'ckpt_step': args.ckpt_step,
            'training_data_path': f'results/{config.system_name}_data/train_{config.n_train}_{config.steps}.pkl',
            'evaluation_data_path': f'results/{config.system_name}_data/val_{config.n_val}_1500.pkl',
        }) 
        config.training_params = ml_collections.ConfigDict({
            'seed': 0,
            'net_name': 'GNS',
            'trial_name': f'{config.n_train}_swish',
            'loss_function': 'lc_state',
            'num_epochs': int(5e2),
            'min_epochs': int(30),
            'batch_size': 5,
            'rollout_timesteps': 1500,
            'log_every_steps': 1,
            'eval_every_steps': 2,
            'ckpt_every_steps': 5,
            'clear_cache_every_steps': 1,
        })
        config.optimizer_params = ml_collections.ConfigDict({
            # 'learning_rate': optax.exponential_decay(init_value=1e-2, transition_steps=1e2, decay_rate=0.1, end_value=1e-5),
            'learning_rate': 1e-3,
        })
        config.net_params = ml_collections.ConfigDict({
            'integration_method': 'euler', 
            'num_mp_steps': 1, # too big causes over-smoothing
            'noise_std': 0.0003,
            'latent_size': 16, # use 32 for 4>= mass spring damper, other <4 use 16
            'hidden_layers': 2,
            'activation': 'swish',
            'use_edge_model': True,
            'layer_norm': True,
            'shared_params': False,
            'dropout_rate': 0.5,
        })

    elif 'mass_spring' in args.system.lower():
        config = ml_collections.ConfigDict({
            'system_name': '2_mass_spring',
            'n_train': 200,
            'steps': 1500,
            'n_val': 20,
            'ckpt_step': args.ckpt_step,
        })
        config.paths = ml_collections.ConfigDict({
            'dir': args.dir,
            'ckpt_step': args.ckpt_step,
            'training_data_path': f'results/{config.system_name}_data/train_{config.n_train}_0.1_0.5_all_random_continuous.pkl',
            'evaluation_data_path': f'results/{config.system_name}_data/val_{config.n_val}_0.1_0.5_passive.pkl',
        }) 
        config.training_params = ml_collections.ConfigDict({
            'seed': 0,
            'net_name': 'GNS',
            'trial_name': f'{config.n_train}',
            'loss_function': 'acceleration',
            'num_epochs': int(5e2),
            'min_epochs': int(30),
            'batch_size': 2,
            'rollout_timesteps': 1500,
            'log_every_steps': 1,
            'eval_every_steps': 2,
            'ckpt_every_steps': 5,
            'clear_cache_every_steps': 1,
            'add_undirected_edges': True,
            'add_self_loops': True,
        })
        config.optimizer_params = ml_collections.ConfigDict({
            # 'learning_rate': optax.exponential_decay(init_value=1e-3, transition_steps=1e2, decay_rate=0.1, end_value=1e-5),
            'learning_rate': 1e-3,
        })
        config.net_params = ml_collections.ConfigDict({
            'prediction': 'acceleration',
            'integration_method': 'euler', 
            'vel_history': 5,
            'control_history': 5,
            'num_mp_steps': 1, # too big causes over-smoothing
            'noise_std': 0.0003,
            'latent_size': 16, # use 32 for 4>= mass spring damper, other <4 use 16
            'hidden_layers': 2,
            'activation': 'relu',
            'use_edge_model': True,
            'layer_norm': True,
            'shared_params': False,
            'dropout_rate': 0.5,
            'add_undirected_edges': config.training_params.add_undirected_edges,
            'add_self_loops': config.training_params.add_self_loops,
        })
    
    else:
        raise NotImplementedError()
    
    return config

def comp_gnn_config(args):
    config = ml_collections.ConfigDict({
        'system_name': 'comp',
    })
    config.paths = ml_collections.ConfigDict({
        'dir': args.dir,
        'ckpt_one_step': 50,
        'ckpt_one_dir': 'results/GNS/LC1/0610-1443_200_new/checkpoint/best_model', 
        'ckpt_two_step': 24,
        'ckpt_two_dir': 'results/GNS/LC2/0610-1926_200_new/checkpoint/best_model', 
        'coupled_lc_data_path': 'results/CoupledLC_data/train_20_1500.pkl',
    })
    config.training_params = ml_collections.ConfigDict({
            'seed': 0,
            'net_name': 'GNS',
            'trial_name': f'{config.system_name}',
            'loss_function': 'lc_state',
            'num_epochs': int(5e2),
            'min_epochs': int(30),
            'batch_size': 5,
            'rollout_timesteps': 1500,
            'log_every_steps': 1,
            'eval_every_steps': 2,
            'ckpt_every_steps': 5,
            'clear_cache_every_steps': 1,
    })
    config.optimizer_params = ml_collections.ConfigDict({
            'learning_rate': 1e-3,
    })
    config.net_params = ml_collections.ConfigDict({
            'integration_method': 'euler', 
            'num_mp_steps': 1,
            'noise_std': 0.0003,
            'latent_size': 16,
            'hidden_layers': 2,
            'activation': 'relu',
            'use_edge_model': True,
            'layer_norm': True,
            'shared_params': False,
            'dropout_rate': 0.5,
    })
    return config

def old_create_comp_gnn_config(args):
    config = ml_collections.ConfigDict()

    config.optimizer_params = ml_collections.ConfigDict({
        'learning_rate': optax.exponential_decay(init_value=1e-3, transition_steps=5e2, decay_rate=0.1, end_value=1e-5),
    })

    config.paths = ml_collections.ConfigDict({
        'dir_one': 'results/GNS/2_mass_spring/0421-1637_2_mass_spring_10000_32',
        'evaluation_data_path_one': 'results/2_mass_spring_data/val_20_0.1_0.5_0_all_random_continuous.pkl',
        'dir_two': 'results/GNS/free_spring/0422-1134_10000_32',
        'evaluation_data_path_two': 'results/free_spring_data/val_20_0.1_0.5_all_random_continuous.pkl',

        'dir': args.dir,
        'training_data_path_comp': 'results/3_mass_spring_data/train_10000_0.1_0.5_0_passive.pkl',
        'evaluation_data_path_comp': 'results/3_mass_spring_data/val_20_0.1_0.5_0_passive.pkl',
    })

    config.training_params = ml_collections.ConfigDict({
        'seed': 0,
        'net_name': 'CompGNS',
        'trial_name': '2+2=3',
        'rollout_timesteps': 1500,
        'loss_function': 'acceleration',
        'num_epochs': int(5e2),
        'batch_size': 2,
        'eval_every_steps': 10,
        'log_every_steps': 1,
        'ckpt_every_steps': 10,
        'clear_cache_every_steps': 1,
    })

    config.net_params_c = ml_collections.ConfigDict({
        'latent_size': 16,
        'hidden_layers': 2,
        'activation': 'relu',
    })

    config.net_params_one = ml_collections.ConfigDict({
        'prediction': 'acceleration',
        'integration_method': 'SemiImplicitEuler', 
        'vel_history': 5,
        'control_history': 5,
        'num_mp_steps': 1, # too big causes oversmoothing
        'noise_std': 0.0003,
        'latent_size': 32,
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
        'latent_size': 32,
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