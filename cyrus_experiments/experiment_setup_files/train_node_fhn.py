import optax
exp_config = {
    'exp_name' : 'train_node_fhn',
    'exp_setup' : {
        'seed' : 1,
    },
    'dataset_setup' : {
        'dataset_type' : 'trajectory_timesteps_in_input',
        'train_dataset_file_name' : 'train.pkl',
        'test_dataset_file_name' : 'val.pkl',
        'dataset_path' : '../environments/fitz_hugh_nagano_data',
    },
    'model_setup' : {
        'model_type' : 'time_control_dependent_node',
        'input_dim': 7, # 6 states + 1 scalar for time.
        'output_dim': 6,
        'dt' : 1e-1,
        'integrator' : 'rk4',
        'network_setup' : {
            'model_type' : 'mlp',
            'input_dim': 9, # 6 states + 1 scalar for time + 2 inputs
            'output_dim': 6,
            'nn_setup_params': {
                'output_sizes': [100, 100, 6],
                'activation': 'relu',
            },
        },
    },
    'trainer_setup' : {
        'trainer_type' : 'sgd',
        'num_training_steps': 100000,
        'minibatch_size': 64,
        'loss_setup' : {
            'loss_function_type' : 'l2_loss',
            'pen_l2_nn_params' : 1e-8,
        },
        'optimizer_setup' : {
            'name' : 'adam',
            'learning_rate' : optax.schedules.cosine_decay_schedule(1e-3,1e5),
        },
    },
}