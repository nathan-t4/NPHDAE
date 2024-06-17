import ml_collections
from time import strftime

def get_optuna_lc_cfg(name, trial) -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict({
        'seed': 0,
        'system_name': name,
        'n_train': 200,
        'steps': 700,
        'n_val': 20,
        'ckpt_step': None,
        'log_every_steps': 1,
        'eval_every_steps': 2,
        'ckpt_every_steps': 5,
        'clear_cache_every_steps': 1,
    })
    config.trial_name = f'{strftime("%m%d-%H%M")}_optuna_{config.n_train}'

    config.paths = ml_collections.ConfigDict({
        'dir': None,
        'ckpt_step': None,
        'training_data_path': f'results/{config.system_name}_data/train_{config.n_train}_{config.steps}.pkl',
        'evaluation_data_path': f'results/{config.system_name}_data/val_{config.n_val}_1500.pkl',
    }) 
    config.training_params = ml_collections.ConfigDict({
        'net_name': 'GNS',
        'loss_function': 'state',
        'num_epochs': int(15),
        'min_epochs': int(15),
        'batch_size': trial.suggest_int('batch_size', 1, 10),
        'rollout_timesteps': 1500,
    })
    config.optimizer_params = ml_collections.ConfigDict({
        # 'learning_rate': optax.exponential_decay(init_value=1e-3, transition_steps=1e2, decay_rate=0.1, end_value=1e-5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    })
    config.net_params = ml_collections.ConfigDict({
        'integration_method': 'euler', 
        'num_mp_steps': trial.suggest_int('num_mp_steps', 1, 5),
        'noise_std': trial.suggest_float('noise_std', 1e-5, 1e-3, log=True),
        'latent_size': trial.suggest_int('latent_size', 4, 16, log=True),
        'hidden_layers': 2,
        'activation': trial.suggest_categorical('activation', ["relu", "swish"]),
        'use_edge_model': True,
        'use_global_model': False,
        'layer_norm': True,
        'shared_params': False,
        'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.8),
    })

    return config