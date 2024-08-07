import jax.numpy as jnp

exp_config = {
    'exp_name' : 'train_phdae_dgu',
    'exp_setup' : {
        'seed' : 1,
    },
    'dataset_setup' : {
        'dataset_type' : 'trajectory_timesteps_in_input',
        'train_dataset_file_name' : 'newest_train_DGU_DAE_2024-08-06-18-39-29.pkl',
        'test_dataset_file_name' : 'newest_test_DGU_DAE_2024-08-06-18-40-00.pkl',
        'dataset_path' : '../environments/dgu_dae_data',
    },
    'model_setup' : {
        'model_type' : 'dgu_phndae',
        'input_dim' : 7, # 6 states + 1 scalar for time.
        'output_dim': 6,
        'dt' : 0.01,
        'AC' : [[0.0], [0.0], [1.0]],
        'AR' : [[1.0], [-1.0], [0.0]],
        'AL' : [[0.0], [1.0], [-1.0]],
        'AV' : [[1.0], [0.0], [0.0]],
        'AI' : [[0.0], [0.0], [-1.0]],
        'H_net_setup': {
            'model_type' : 'mlp',
            'input_dim' : 1,
            'output_dim': 1,
            'nn_setup_params': {
                'output_sizes': [32, 32, 1],
                'activation': 'tanh',
            },
        },
        'r_net_setup' : {
            'model_type' : 'mlp',
            'input_dim' : 1,
            'output_dim': 1,
            'nn_setup_params': {
                'output_sizes': [32, 32, 1],
                'activation': 'relu',
            },
        },
        'q_net_setup' : {
            'model_type' : 'mlp',
            'input_dim' : 1,
            'output_dim': 1,
            'nn_setup_params': {
                'output_sizes': [32, 32, 1],
                'activation': 'relu',
            },
        },
        'u_func_freq' : 0.0,
        'u_func_current_source_magnitude' : 0.1,
        'u_func_voltage_source_magnitude' : 1.0,
    },
    'trainer_setup' : {
        'trainer_type' : 'sgd',
        'num_training_steps': 30000,
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
        },
    },
}