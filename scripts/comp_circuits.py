import os
import jax
import optax
from flax.training.train_state import TrainState
import orbax.checkpoint as ocp

from scripts.model_instances.comp_nets import CompPHGNS
from configs.comp_circuits_old import get_comp_gnn_config
from helpers.graph_builder_factory import gb_factory
from utils.train_utils import *
from utils.gnn_utils import *
from utils.comp_utils import *

def compose(config):
    subsystem_names = config.subsystem_names
    known_subsystem = config.known_subsystem
    num_subsystems = len(subsystem_names)
    net_params = config.net_params
    optimizer_params = config.optimizer_params
    paths = config.paths
    ckpt_dirs = paths.ckpt_dirs
    ckpt_steps = paths.ckpt_steps
    Alambda = config.Alambda # TODO

    if paths.dir == None:
        config.paths.dir = os.path.join(os.curdir, f'results/CompGNN/{config.trial_name}')
        paths.dir = config.paths.dir

    plot_dir = os.path.join(paths.dir, 'eval_plots')

    rng = jax.random.key(config.seed)
    rng, init_rng, net_rng = jax.random.split(rng, 3)

    graph_builder = gb_factory('') # Get TestGraphBuilder

    eval_gb = graph_builder(paths.comp_data_path)

    init_graph = eval_gb.get_graph(traj_idx=0, t=0)
    init_graphs = explicit_unbatch_graph(init_graph, system_configs)

    init_control = eval_gb._control[0,0]
    init_controls = explicit_unbatch_control(init_control, system_configs) # TODO

    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)  

    nets = []
    params = []
    txs = []
    train_states = []
    gbs = []
    graph_to_state = []
    state_to_graph = []
    incidence_matrices = []
    system_configs = []

    for i in range(num_subsystems):
        net_params[i].training = False 

        gb = gb_factory(subsystem_names[i])(paths.training_data_paths[i])
        gbs.append(gb)

        g_to_s = gb.get_state_from_graph
        s_to_g = gb.get_graph_from_state
        
        graph_to_state.append(g_to_s)
        state_to_graph.append(s_to_g)

        AC = config.incidence_matrices[i].AC
        AR = config.incidence_matrices[i].AR
        AL = config.incidence_matrices[i].AL
        AV = config.incidence_matrices[i].AV
        AI = config.incidence_matrices[i].AI

        incidence_matrices.append((AC, AR, AL, AV, AI))

        if not known_subsystem[i]:
            net = create_net(subsystem_names[i], net_params[i])
            net.graph_from_state = g_to_s,
            net.state_from_graph = s_to_g,
            net.edge_idxs = gb.edge_idxs
            net.node_idxs = gb.node_idxs
            net.include_idxs = gb.include_idxs
            nets.append(net)

            param = net.init(init_rng, init_graphs[i], init_controls[i], net_rng)
            params.append(param)

            tx = optax.adam(**optimizer_params[i])
            txs.append(tx)

            ckpt_mngr = ocp.CheckpointManager(
                os.path.abspath(ckpt_dirs[i]),
                options=options,
                item_handlers=ocp.StandardCheckpointHandler(),
            )

            train_state = TrainState.create(
                apply_fn=net.apply,
                params=param,
                tx=tx
            )
            # Restore state from checkpoint
            train_state = ckpt_mngr.restore(ckpt_steps[i], args=ocp.args.StandardRestore(train_state))
            train_states.append(train_state)
        
            system_config = get_system_config(*incidence_matrices[i])
            system_configs.append(system_config)

        else:
            system_config = get_system_config(*incidence_matrices[i])
            system_config['is_last'] = True
        
        system_configs.append(system_config)

    assert all([net.dt == nets[0].dt for net in nets])
    assert all([net.T == nets[0].T for net in nets])
    assert all([net.integration_method == nets[0].integration_method for net in nets])
    dt = nets[0].dt
    T = nets[0].T
    integration_method = nets[0].integration_method

    # Initialize composite GNS
    comp_net_config = {
        'integration_method': integration_method,
        'dt': dt,
        'T': T,
        'train_states': train_states,
        'graph_to_state': graph_to_state,
        'state_to_graph': state_to_graph,
        'system_configs': system_configs,
        'Alambda': Alambda,
    }

    comp_net = CompPHGNS(**comp_net_config)
    comp_params = net.init(init_rng, init_graph, init_control, net_rng)
    comp_tx = optax.adam(1e-3)

    state = TrainState.create(
        apply_fn=comp_net.apply,
        params=comp_params,
        tx=comp_tx,
    )

    def rollout(state, traj_idx, ti = 0):
        tf = jnp.floor_divide(config.rollout_timesteps + 1, net.T)
        tf_idxs = ti + jnp.arange(1, tf)
        tf_idxs = tf_idxs * T
        t0_idxs = tf_idxs - T
        ts = tf_idxs * net.dt
        graph = eval_gb.get_graph(traj_idx, ti)
        controls = eval_gb.get_control(traj_idx, t0_idxs)
        exp_data = eval_gb.get_exp_data(traj_idx, tf_idxs)
        get_pred_data = eval_gb.get_pred_data
        get_batch_pred_data = eval_gb.get_batch_pred_data
        
        def forward_pass(carry, inputs):  
            graph, u_hats = carry
            control, t = inputs         
            graphs, u_hats = state.apply_fn(state.params, graph, control, u_hats, t, jax.random.key(config.seed))
            pred_data = get_batch_pred_data(graphs)
            graphs = [graph._replace(globals=None) for graph in graphs]
            return (graphs, u_hats), pred_data

        init_u_hat = None
        init_timesteps = t0_idxs * dt
        _, pred_data = jax.lax.scan(forward_pass, (graph, init_u_hat), (controls, init_timesteps))
        
        losses = [
            jnp.sum(optax.l2_loss(predictions=pred_data[i], targets=exp_data[i])) for i in range(len(exp_data))
        ]
        eval_metrics = [EvalMetrics.single_from_model_output(loss=loss) for loss in losses]
        pred_data = np.concatenate(pred_data, axis=1)
        exp_data = np.concatenate(exp_data, axis=1)
        return pred_data, exp_data, eval_metrics
    
    print('##################################################')
    print(f"Evaluating composition of subsystems {subsystem_names}")
    print(f"The known subsystems are {subsystem_names[[i for i, b in enumerate(known_subsystem) if b]]}")
    print(f"Saving plots to {os.path.relpath(plot_dir)}")
    print('##################################################')

    error_sums = [0] * 4
    for i in range(eval_gb._num_trajectories): # eval_gb._num_trajectories
        pred_data, exp_data, eval_metrics = rollout(state, traj_idx=i)
        for j in range(len(error_sums)):
            error_sums[j] += eval_metrics[j].compute()['loss']

        eval_gb.plot(pred_data, exp_data, 
                     plot_dir=os.path.join(plot_dir['plot'], f'traj_{i}'),
                     prefix=f'comp_eval_traj_{i}')

    rollout_mean_error = np.array(error_sums) / (eval_gb._num_trajectories * eval_gb._num_timesteps)

    print('##################################################')
    print("Finished evaluating composition")
    print("\nMean rollout stats:")
    print(f"\tDifferential states error: {rollout_mean_error[0]}")
    print(f"\tAlgebraic states error: {rollout_mean_error[1]}")
    print(f"\tHamiltonian error: {rollout_mean_error[2]}")
    print(f"\tResidual: {rollout_mean_error[3]}")
    print('##################################################')

    metrics = {
        'subsystem_names': subsystem_names,
        'known_subsystem': known_subsystem,
        'ckpt_dirs': ckpt_dirs,
        'rollout_mean_error_states': rollout_mean_error.tolist(),
        'rollout_mean_error': str(rollout_mean_error.mean()),
    }
    with open(os.path.join(paths.dir, 'metrics.js'), "w") as outfile:
        json.dump(metrics, outfile, indent=4)

    return metrics['rollout_mean_error']
    
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default=None)
    args = parser.parse_args()
    cfg = get_comp_gnn_config(args)
    compose(cfg)