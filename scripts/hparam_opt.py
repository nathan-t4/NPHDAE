"""
    Tune hyper-parameters using optuna

    See configs/optuna_lc.py to see the specific hyper-parameters being tuned.
"""
import optuna
import jax
from train_gnn import train
from configs.optuna_lc import get_optuna_lc_cfg

def optimize(system_name):
    def objective(trial):
        cfg = get_optuna_lc_cfg(system_name, trial)
        loss = train(cfg, trial)
        jax.clear_caches()
        return loss

    study = optuna.create_study(
        study_name=f'{system_name}',
        direction='minimize',
        # storage=f'sqlite:///{system_name}/optuna_hparam_search.db',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=15),
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=10-len(study.trials), n_jobs=1)

    trial = study.best_trial
    print(f'Best Error: {trial.value:4.2%}')
    print(f'Best Params:')
    for key, value in trial.params.items():
        print(f'-> {key}: {value}')

    fig = optuna.visualization.plot_intermediate_values(study)
    fig.show()

    fig = optuna.visualization.plot_param_importances(study)
    fig.show()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--system', type=str, required=True)
    args = parser.parse_args()

    compatible_systems = ['LC1', 'LC2', 'CoupledLC']

    assert args.system in compatible_systems, 'Invalid system name'

    optimize(args.system)
