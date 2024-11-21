import optax
import jax.numpy as jnp
import haiku as hk

exp_config = {
    'exp_name' : 'baseline',
    'exp_setup' : {
        'seed' : 0,
    },
    'dataset_setup' : {
        'dataset_type' : 'trajectory_timesteps_in_input',
        'train_dataset_file_name' : 'train_less_data.pkl',
        'test_dataset_file_name' : 'val.pkl',
        'dataset_path' : '../environments/fitz_hugh_nagano_data',
    },
    'model_setup' : {
        'model_type' : 'dgu_phndae',
        'input_dim' : 7, # 6 states + 1 scalar for time.
        'output_dim': 6,
        'dt': 1e-1,
        'one_timestep_solver': 'rk4',
        'AC' : [[1.0], [0.0], [0.0]],
        'AR' : [[1.0, -1.0],
                [0.0, 1.0],
                [0.0, 0.0]],
        'AL' : [[0.0], [0.0], [-1.0]],
        'AV' : [[0.0], [-1.0], [1.0]],
        'AI' : [[1.0], [0.0], [0.0]],
        'R': 1.0, # not used
        'L': 1.0, # not used
        'C': 1.0, # not used
        'scalings': [1,1,1,1,1],
        'H_net_setup': {
            'model_type' : 'mlp',
            'input_dim' : 1,
            'output_dim': 1,
            'nn_setup_params': {
                'output_sizes': [100, 100, 1],
                'activation': 'tanh',
                # 'w_init': hk.initializers.TruncatedNormal(stddev=1/2.9e-1),
            },
        },
        'r_net_setup' : {
            'model_type' : 'mlp',
            'input_dim' : 2,
            'output_dim': 2,
            'nn_setup_params': {
                'output_sizes': [100, 100, 2],
                'activation': 'relu',
                # 'w_init': hk.initializers.TruncatedNormal(stddev=1/5),
            },
        },
        'q_net_setup' : {
            'model_type' : 'mlp',
            'input_dim' : 1,
            'output_dim': 1,
            'nn_setup_params': {
                'output_sizes': [100, 100, 1],
                'activation': 'relu',
                # 'w_init': hk.initializers.TruncatedNormal(stddev=1/5),
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
            # 'learning_rate' : 1e-5,
            'learning_rate' : optax.schedules.cosine_decay_schedule(1e-4,1e5),
            # 'weight_decay': 1e-2,
            'clipping': 1.0,
        },
    },
}