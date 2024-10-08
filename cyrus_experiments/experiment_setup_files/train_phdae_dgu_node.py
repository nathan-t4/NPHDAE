import jax.numpy as jnp

exp_config = {
    'exp_name' : 'train_phdae_dgu_node',
    'exp_setup' : {
        'seed' : 1,
    },
    'dataset_setup' : {
        'dataset_type' : 'trajectory_timesteps_in_input',
        'train_dataset_file_name' : 'train_500_1000_real.pkl',
        'test_dataset_file_name' : 'val_20_800_real.pkl',
        # 'train_dataset_file_name' : 'train_1e-5.pkl',
        # 'test_dataset_file_name' : 'val_1e-5.pkl',
        'dataset_path' : '../environments/dgu_dae_data',
    },
    'model_setup' : {
        'model_type' : 'dgu_phndae',
        'input_dim' : 7, # 6 states + 1 scalar for time.
        'output_dim': 6,
        'dt' : 1e-4,
        # 'dt' : 1e-5,
        'AC' : [[0.0], [0.0], [1.0]],
        'AR' : [[-1.0], [1.0], [0.0]],
        'AL' : [[0.0], [1.0], [-1.0]],
        'AV' : [[1.0], [0.0], [0.0]],
        'AI' : [[0.0], [0.0], [-1.0]],
        'R': 0.2,
        'L': 1.8e-3,
        'C': 2.2e-3,
        'H_net_setup': {
            'model_type' : 'time_dependent_node',
            'input_dim' : 1,
            'output_dim': 1,
            'dt': 1e-4,
            'nn_setup_params': {
                'output_sizes': [32, 32, 1],
                'activation': 'tanh',
            },
        },
        'r_net_setup' : {
            'model_type' : 'time_dependent_node',
            'input_dim' : 1,
            'output_dim': 1,
            'dt': 1e-4,
            'nn_setup_params': {
                'output_sizes': [32, 32, 1],
                'activation': 'relu',
            },
        },
        'q_net_setup' : {
            'model_type' : 'time_dependent_node',
            'input_dim' : 1,
            'output_dim': 1,
            'dt': 1e-4,
            'nn_setup_params': {
                'output_sizes': [32, 32, 1],
                'activation': 'relu',
            },
        },
        'u_func_freq' : None,
        'u_func_current_source_magnitude' : 1.0,
        'u_func_voltage_source_magnitude' : 100.0,
    },
    'trainer_setup' : {
        'trainer_type' : 'sgd',
        'num_training_steps': 70000,
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