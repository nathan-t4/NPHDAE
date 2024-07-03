import os
from scripts.train_gnn import train
from scripts.comp_circuits import compose
from scripts.reuse_model import transfer
from time import strftime
from argparse import ArgumentParser
from helpers.config_factory import config_factory

def run_train():
    pass

def run_composition(args, trial_names, ckpt_steps):
    comp_errors = []   
    for i in range(5):
        config = config_factory('CompCircuits', args)
        config.trial_name = f'{strftime("%m%d")}_T={i + 1}'
        subsystem_config = config_factory('LC1', args)
        subsystem_config.net_params.T = i + 1

        config.paths.ckpt_one_dir = trial_names[i]
        config.paths.ckpt_one_step = ckpt_steps[i]
        config.paths.ckpt_two_dir = trial_names[i]
        config.paths.ckpt_two_step = ckpt_steps[i]

        config.optimizer_params_1 = subsystem_config.optimizer_params
        config.net_params_1 = subsystem_config.net_params

        config.optimizer_params_2 = subsystem_config.optimizer_params
        config.net_params_2 = subsystem_config.net_params

        error = compose(config)
        comp_errors.append(error)

    print(f'Composition errors {comp_errors}')
    min_comp_error = min(comp_errors)
    print(f'Minimum error is {min_comp_error} when T = {comp_errors.index(min_comp_error) + 1}')

def run_transfer(args, trial_names, ckpt_steps):
    transfer_errors = []
    for i in range(5):
        config = config_factory('ReuseModel', args)
        config.trial_name = f'{strftime("%m%d")}_T={i + 1}'
        subsystem_config = config_factory('LC1', args)
        subsystem_config.net_params.T = i + 1

        config.paths.ckpt_one_dir = trial_names[i]
        config.paths.ckpt_one_step = ckpt_steps[i]

        config.training_params_1 = subsystem_config.training_params
        config.optimizer_params_1 = subsystem_config.optimizer_params
        config.net_params_1 = subsystem_config.net_params

        config.set_nodes = False

        error = transfer(config)
        transfer_errors.append(error)

    print(f'Transfer errors {transfer_errors}')
    min_transfer_error = min(transfer_errors)
    print(f'Minimum error is {min_transfer_error} when T = {transfer_errors.index(min_transfer_error) + 1}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--ckpt_step', type=int, default=None)
    parser.add_argument('--system', type=str, required=True)
    args = parser.parse_args()

    trial_names = []
    errors = []
    ckpt_steps = []

    # for i in range(5):
    #     print(f'Training with T = {i + 1}')
    #     config = config_factory(args.system, args)
    #     config.net_params.T = i + 1
    #     config.trial_name =  f'{strftime("%m%d")}_T={config.net_params.T}_learn_nodes_2'

    #     final_config = train(config)
    #     trial_names.append(final_config.paths.dir)
    #     ckpt_steps.append(final_config.paths.ckpt_step)
    #     errors.append(final_config.metrics.min_error)

    # min_error = min(errors)
    # print(f'Rollout errors {errors}')
    # print(f'Minimum error is {min_error} when T = {errors.index(min_error) + 1}')

   
    # trial_names = [os.path.join(path, "\checkpoint\best_model") for path in trial_names]

    trial_names = [f'results/GNS/LC1/0627_T={i + 1}_learn_nodes/checkpoint/best_model/' for i in range(5)]

    ckpt_steps = [38, 44, 40, 44, 34]

    if args.system == 'LC1':
        # run_composition(args, trial_names, ckpt_steps)
        run_transfer(args, trial_names, ckpt_steps)