import ml_collections       
import optax

def get_mass_spring_config(args):
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
        'latent_size': 16,
        'hidden_layers': 2,
        'activation': 'relu',
        'use_edge_model': True,
        'use_global_model': False,
        'layer_norm': True,
        'shared_params': False,
        'dropout_rate': 0.5,
        'add_undirected_edges': config.training_params.add_undirected_edges,
        'add_self_loops': config.training_params.add_self_loops,
    })
    return config