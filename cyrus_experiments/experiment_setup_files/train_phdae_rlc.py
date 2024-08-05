import jax.numpy as jnp

exp_config = {
    'exp_name' : 'train_phdae_rlc',
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
        'model_type' : 'phndae',
        'input_dim' : 7, # 6 states + 1 scalar for time.
        'output_dim': 6,
        'dt' : 0.01,
        'AC' : [[0.0], [0.0], [1.0]],
        'AR' : [[1.0], [-1.0], [0.0]],
        'AL' : [[0.0], [1.0], [-1.0]],
        'AV' : [[1.0], [0.0], [0.0]],
        'AI' : [[0.0], [0.0], [0.0]],
        'H_net_setup': {
            'model_type' : 'mlp',
            'input_dim' : 1,
            'output_dim': 1,
            'nn_setup_params': {
                'output_sizes': [32, 32, 1],
                'activation': 'tanh',
            },
        },
        'u_func' : lambda t, params : [jnp.sin(30 * t)],
    },
    'trainer_setup' : {
        'trainer_type' : 'sgd',
        'num_training_steps': 5000,
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