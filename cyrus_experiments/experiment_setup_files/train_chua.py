import jax.numpy as jnp
import haiku as hk

exp_config = {
    'exp_name' : 'chua',
    'exp_setup' : {
        'seed' : 1,
    },
    'dataset_setup' : {
        'dataset_type' : 'trajectory_timesteps_in_input',
        'train_dataset_file_name' : 'train.pkl',
        'test_dataset_file_name' : 'val.pkl',
        'dataset_path' : '../environments/chua_data',
    },
    'model_setup' : {
        'model_type' : 'dgu_phndae',
        'input_dim' : 5, # 6 states + 1 scalar for time.
        'output_dim': 4,
        'dt': 1e-3,
        'regularization_method': 'none',
        'reg_param': 0.0,
        'one_timestep_solver': 'rk4',
        'AC' : [[0.0, 1.0],
                [1.0, 0.0]],
        'AR' : [[-1.0, 0.0],
                [1.0, 1.0]],
        'AL' : [[-1.0], [0.0]],
        'AV' : [[0.0], [0.0]],
        'AI' : [[0.0], [0.0]],
        'R': 1.0, # not used
        'L': 1.0, # not used
        'C': 1.0, # not used
        'scalings': [1,1,1,1,1],
        'H_net_setup': {
            'model_type' : 'mlp',
            'input_dim' : 1,
            'output_dim': 1,
            'use_batch_norm': False,
            'nn_setup_params': {
                'output_sizes': [32, 32, 1],
                'activation': 'tanh',
                # 'w_init': hk.initializers.TruncatedNormal(stddev=1/2.9e-1),
            },
        },
        'r_net_setup' : {
            'model_type' : 'mlp',
            'input_dim' : 2,
            'output_dim': 2,
            'use_batch_norm': False,
            'nn_setup_params': {
                'output_sizes': [32, 32, 2],
                'activation': 'relu',
                # 'w_init': hk.initializers.TruncatedNormal(stddev=1/5),
            },
        },
        'q_net_setup' : {
            'model_type' : 'mlp',
            'input_dim' : 2,
            'output_dim': 2,
            'use_batch_norm': False,
            'nn_setup_params': {
                'output_sizes': [32, 32, 2],
                'activation': 'relu',
                # 'w_init': hk.initializers.TruncatedNormal(stddev=1/5),
            },
        },
        'u_func_current_frequency': None,
        'u_func_current_source_magnitude' : None,
        'u_func_voltage_frequency': None,
        'u_func_voltage_source_magnitude' : None,
    },
    'trainer_setup' : {
        'trainer_type' : 'sgd',
        'num_training_steps': 30000,
        'minibatch_size': 8,
        'loss_setup' : {
            'loss_function_type' : 'l2_and_g_loss',
            'pen_l2' : 1.0,
            'pen_g' : 1e-1,
            'pen_l2_nn_params' : 1e-8,
        },
        'optimizer_setup' : {
            'name' : 'adam',
            'learning_rate' : 1e-4,
            'clipping': 1.0,
        },
    },
}