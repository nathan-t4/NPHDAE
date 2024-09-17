import jax.numpy as jnp

exp_config = {
    'exp_name' : 'train_phdae_dgu',
    'exp_setup' : {
        'seed' : 1,
    },
    'dataset_setup' : {
        'dataset_type' : 'trajectory_timesteps_in_input',
        'train_dataset_file_name' : 'train_500_1000_random.pkl',
        'test_dataset_file_name' : 'val_20_800_random.pkl',
        'dataset_path' : '../environments/dgu_dae_data',
    },
    'model_setup' : {
        'model_type' : 'dgu_phndae',
        'input_dim' : 7, # 6 states + 1 scalar for time.
        'output_dim': 6,
        'dt' : 1e-4,
        'AC' : [[0.0], [0.0], [1.0]],
        'AR' : [[1.0], [-1.0], [0.0]],
        'AL' : [[0.0], [1.0], [-1.0]],
        'AV' : [[1.0], [0.0], [0.0]],
        'AI' : [[0.0], [0.0], [-1.0]],
        'R': 0.2,
        'L': 1.8e-3,
        'C': 2.2e-3,
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
            'input_dim' : 1, # TODO: should take difference in e1 and e2
            'output_dim': 1,
            'nn_setup_params': {
                'output_sizes': [32, 32, 1],
                'activation': 'relu',
            },
        },
        'q_net_setup' : {
            'model_type' : 'mlp',
            'input_dim' : 1, # TODO: should take in e3
            'output_dim': 1,
            'nn_setup_params': {
                'output_sizes': [32, 32, 1],
                'activation': 'relu',
            },
        },
        'u_func_freq' : 0.0,
        'u_func_current_source_magnitude' : 0.1,
        'u_func_voltage_source_magnitude' : 100.0,
    },
    'trainer_setup' : {
        'trainer_type' : 'sgd',
        'num_training_steps': 30000,
        'minibatch_size': 128,
        'loss_setup' : {
            'loss_function_type' : 'l2_and_g_loss',
            'pen_l2' : 1,
            'pen_g' : 1e-2,
            'pen_l2_nn_params' : 0.0,
        },
        'optimizer_setup' : {
            'name' : 'adam',
            'learning_rate' : 1e-4,
        },
    },
}