exp_config = {
    'exp_name' : 'train_mlp_rlc',
    'exp_setup' : {
        'seed' : 1,
    },
    'dataset_setup' : {
        'dataset_type' : 'trajectory_timesteps_in_input',
        'train_dataset_file_name' : 'train_RLC_DAE_2024-08-04-18-31-33.pkl',
        'test_dataset_file_name' : 'test_RLC_DAE_2024-08-04-18-32-05.pkl',
        'dataset_path' : '../environments/rlc_dae_data',
        'num_training_trajectories' : 100,
        'num_testing_trajectories' : 20,
    },
    'model_setup' : {
        'model_type' : 'mlp',
        'input_dim': 7, # 6 states + 1 scalar for time.
        'output_dim': 6,
        'nn_setup_params': {
            'output_sizes': [64, 64, 6],
            'activation': 'relu'
        }
    },
    'trainer_setup' : {
        'trainer_type' : 'sgd',
        'num_training_steps': 20000,
        'minibatch_size': 32,
        'loss_setup' : {
            'loss_function_type' : 'l2_loss',
            'pen_l2_nn_params' : 1e-8,
        },
        'optimizer_setup' : {
            'name' : 'adam',
            'learning_rate' : 1e-4,
        },
    },
}