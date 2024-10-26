import jax.numpy as jnp
import haiku as hk

exp_config = {
    'exp_name' : 'phdae_dgu_user101',
    'exp_setup' : {
        'seed' : 1,
    },
    'dataset_setup' : {
        'dataset_type' : 'trajectory_timesteps_in_input',
        'train_dataset_file_name' : 'train_1e-5.pkl',
        'test_dataset_file_name' : 'val_1e-5.pkl',
        'dataset_path' : '../environments/dgu_dae_data',
    },
    'model_setup' : {
        'model_type' : 'dgu_phndae',
        'input_dim' : 7, # 6 states + 1 scalar for time.
        'output_dim': 6,
        'dt': 1e-5,
        'regularization_method': 'none',
        'reg_param': 0.0,
        'one_timestep_solver': 'rk4',
        'AC' : [[0.0], [0.0], [1.0]],
        'AR' : [[-1.0], [1.0], [0.0]],
        'AL' : [[0.0], [1.0], [-1.0]],
        'AV' : [[1.0], [0.0], [0.0]],
        'AI' : [[0.0], [0.0], [-1.0]],
        'R': 0.2,
        'L': 1.8e-3,
        'C': 2.2e-3,
        # 'scalings': [2.3627788e-01, 3.3838414e-02, 1.0000000e+02, 9.6240173e+01, 1.0739904e+02, -1.8799120e+01], # means
        # 'scalings': [8.12288693e+00, 9.54885663e+00, 1.00000000e-05, 8.59397056e-02, 1.78703497e-02, 1.71879426e-02], # std
        # 'scalings': [2.6484454 , 3.4440985 , 1e-5, 0.03099686, 0.00582658, 0.00619937], # max-min
        'scalings': [10, 10, 1, 1, 1, 1], # user specified
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
            'input_dim' : 1,
            'output_dim': 1,
            'use_batch_norm': False,
            'nn_setup_params': {
                'output_sizes': [32, 32, 1],
                'activation': 'relu', # relu
                # 'w_init': hk.initializers.TruncatedNormal(stddev=1/5),
            },
        },
        'q_net_setup' : {
            'model_type' : 'mlp',
            'input_dim' : 1,
            'output_dim': 1,
            'use_batch_norm': False, # TODO
            'nn_setup_params': {
                'output_sizes': [32, 32, 1],
                'activation': 'relu', # relu
                # 'w_init': hk.initializers.TruncatedNormal(stddev=1/5),
            },
        },
        'u_func_freq' : None,
        'u_func_current_source_magnitude' : 1.0,
        'u_func_voltage_source_magnitude' : 100.0,
    },
    'trainer_setup' : {
        'trainer_type' : 'sgd',
        'num_training_steps': 300000, # try 100000 with rk4
        'minibatch_size': 128,
        'loss_setup' : {
            'loss_function_type' : 'l2_and_g_loss',
            'pen_l2' : 1,
            'pen_g' : 1e-2,
            'pen_l2_nn_params' : 1e-8,
        },
        'optimizer_setup' : {
            'name' : 'adam',
            'learning_rate' : 1e-4,
            # 'weight_decay': 1e-2,
            'clipping': 1.0,
        },
    },
}