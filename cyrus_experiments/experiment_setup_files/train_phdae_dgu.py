import jax.numpy as jnp
import haiku as hk
import optax

exp_config = {
    'exp_name' : 'baseline',
    'exp_setup' : {
        'seed' : 0,
    },
    'dataset_setup' : {
        'dataset_type' : 'trajectory_timesteps_in_input',
        'train_dataset_file_name' : 'train.pkl',
        'test_dataset_file_name' : 'val.pkl',
        'dataset_path' : '../environments/dgu_dae_data',
    },
    'model_setup' : {
        'model_type' : 'dgu_phndae',
        'input_dim' : 7, # 6 states + 1 scalar for time.
        'output_dim': 6,
        'dt': 1e-2,
        'one_timestep_solver': 'rk4',
        'AC' : [[0.0], [0.0], [1.0]],
        'AR' : [[-1.0], [1.0], [0.0]],
        'AL' : [[0.0], [1.0], [-1.0]],
        'AV' : [[1.0], [0.0], [0.0]],
        'AI' : [[0.0], [0.0], [-1.0]],
        'R': 1.2,
        'L': 1.8,
        'C': 2.2,
        'H_net_setup': {
            'model_type' : 'mlp',
            'input_dim' : 1,
            'output_dim': 1,
            'nn_setup_params': {
                'output_sizes': [100, 100, 1],
                'activation': 'tanh',
            },
        },
        'r_net_setup' : {
            'model_type' : 'mlp',
            'input_dim' : 1,
            'output_dim': 1,
            'nn_setup_params': {
                'output_sizes': [100, 100, 1],
                'activation': 'relu',
            },
        },
        'q_net_setup' : {
            'model_type' : 'mlp',
            'input_dim' : 1,
            'output_dim': 1,
            'nn_setup_params': {
                'output_sizes': [100, 100, 1],
                'activation': 'relu',
            },
        },
    },
    'trainer_setup' : {
        'trainer_type' : 'sgd',
        'num_training_steps': 100000,
        'minibatch_size': 128,
        'loss_setup' : {
            'loss_function_type' : 'l2_and_g_loss',
            'pen_l2' : 1.0,
            'pen_g' : 1e-2,
            'pen_l2_nn_params' : 1e-8,
        },
        'optimizer_setup' : {
            'name' : 'adam',
            'learning_rate' : optax.schedules.cosine_decay_schedule(1e-4,1e5),
            'clipping': 1.0,
        },
    },
}