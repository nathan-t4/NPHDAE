from scripts.train import *

# default_validation_dataset_path = 'results/double_mass_spring_data/2sin_val.pkl'
default_validation_dataset_path = 'results/switched_double_mass_spring_data/1_switch_5uniform_val.pkl'

def eval(args):
    rng = jax.random.key(0)
    rng, init_rng = jax.random.split(rng)
    init_graph = build_graph(dataset_path=args.data, key=init_rng, batch_size=1)[0]
    work_dir = args.dir
    plot_dir = os.path.join(work_dir, 'plots')

    def load_params(path: str):
        with open(path) as f:
            run_params = json.load(f)
        return run_params
    
    run_params_path = os.path.join(work_dir, f'run_params.js')
    run_params = load_params(run_params_path)
    net_params = run_params['net_params']
    training_params = run_params['training_params']

    tx = optax.adam(learning_rate=training_params['lr'])

    val_net = GraphNet(**net_params)
    params = jax.jit(val_net.init)(init_rng, init_graph)
    # params = val_net.init(init_rng, init_graph)
    state = train_state.TrainState.create(
        apply_fn=val_net.apply, params=params, tx=tx,
    )

    checkpoint_dir = os.path.join(work_dir, 'checkpoints')
    # Setup checkpointing for model
    ckpt = checkpoint.Checkpoint(checkpoint_dir)
    state = ckpt.restore_or_initialize(state)
    

    rng, val_rng = jax.random.split(rng)
    val_state = state.replace(apply_fn=val_net.apply)
    val_graphs = build_graph(dataset_path=args.data,
                             key=val_rng,
                             batch_size=10)
    val_metrics, pred_qs, exp_qs, pred_as, exp_as = eval_model(val_state, val_graphs)
    save_evaluation_curves(plot_dir, 'position_val', pred_qs, exp_qs)
    save_evaluation_curves(plot_dir, 'acceleration_val', pred_as, exp_as)

    print(f"Validation loss {round(val_metrics.compute()['loss'],4)}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--data', type=str, default=default_validation_dataset_path)
    args = parser.parse_args()

    eval(args)