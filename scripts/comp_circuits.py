import os
import jax
import optax
from flax.training.train_state import TrainState
import orbax.checkpoint as ocp

from scripts.model_instances.comp_nets import CompLCGNS
from configs.comp_circuits import get_comp_gnn_config
from helpers.graph_builder_factory import gb_factory
from utils.train_utils import *
from utils.gnn_utils import *
from utils.comp_utils import *

def compose(config):
    net_params_1 = config.net_params_1
    net_params_2 = config.net_params_2
    paths = config.paths

    if paths.dir == None:
        config.paths.dir = os.path.join(os.curdir, f'results/CompGNN/{config.trial_name}')
        paths.dir = config.paths.dir

    plot_dir = os.path.join(paths.dir, 'eval_plots')

    rng = jax.random.key(config.seed)
    rng, init_rng, net_rng = jax.random.split(rng, 3)

    graph_builder = gb_factory('CoupledLC')

    eval_gb = graph_builder(paths.coupled_lc_data_path)

    net_params_1.training = False
    net_params_2.training = False

    net_one = create_net(config.net_one_name, net_params_1)
    net_two = create_net(config.net_two_name, net_params_2)

    gb_one = gb_factory(config.net_one_name)(paths.training_data_one)
    gb_two = gb_factory(config.net_two_name)(paths.training_data_two)

    net_one.graph_from_state = gb_one.get_graph_from_state
    net_two.graph_from_state = gb_two.get_graph_from_state

    net_one.edge_idxs = gb_one.edge_idxs
    net_one.node_idxs = gb_one.node_idxs
    net_two.edge_idxs = gb_two.edge_idxs
    net_two.node_idxs = gb_two.node_idxs
    net_two.include_idxs = np.array([0,1])

    system_one_config = get_system_config(gb_one.get_graph(0, 0))
    system_two_config = get_system_config(gb_two.get_graph(0, 0))

    init_graph = eval_gb.get_graph(traj_idx=0, t=0)
    init_graph_one, init_graph_two = explicit_unbatch_graph(init_graph, system_one_config, system_two_config)

    init_control = eval_gb._control[0,0]
    init_control_one = init_control[:system_one_config['num_volt_sources']+system_one_config['num_cur_sources']]
    init_control_two = init_control[system_one_config['num_volt_sources']+system_one_config['num_cur_sources']:]

    params_one = net_one.init(init_rng, init_graph_one, init_control_one, net_rng)

    params_two = net_two.init(init_rng, init_graph_two, init_control_two, net_rng)

    tx_one = optax.adam(**config.optimizer_params_1)

    tx_two = optax.adam(**config.optimizer_params_2)

    state_one = TrainState.create(
        apply_fn=net_one.apply,
        params=params_one,
        tx=tx_one,
    )

    state_two = TrainState.create(
        apply_fn=net_two.apply,
        params=params_two,
        tx=tx_two,
    )

    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)  
    paths.ckpt_one_dir = os.path.abspath(paths.ckpt_one_dir)  
    ckpt_mngr_one = ocp.CheckpointManager(
        paths.ckpt_one_dir,
        options=options,
        item_handlers=ocp.StandardCheckpointHandler(),
    )
    paths.ckpt_two_dir = os.path.abspath(paths.ckpt_two_dir)  
    ckpt_mngr_two = ocp.CheckpointManager(
        paths.ckpt_two_dir,
        options=options,
        item_handlers=ocp.StandardCheckpointHandler(),
    )

    # Restore state from checkpoint
    state_one = ckpt_mngr_one.restore(paths.ckpt_one_step, args=ocp.args.StandardRestore(state_one))
    state_two = ckpt_mngr_two.restore(paths.ckpt_two_step, args=ocp.args.StandardRestore(state_two))

    assert net_one.dt == net_two.dt
    assert net_one.T == net_two.T
    assert net_one.integration_method == net_two.integration_method
    dt = net_one.dt
    T = net_one.T
    integrator = net_one.integration_method

    # Initialize composite GNS
    comp_net_config = {
        'integration_method': integrator,
        'dt': dt,
        'T': T,
        'state_one': state_one,
        'state_two': state_two,
        'graph_to_state_one': gb_one.graph_to_state,
        'graph_to_state_two': gb_two.graph_to_state,
        'state_to_graph_one': gb_one.state_to_graph,
        'state_to_graph_two': gb_two.state_to_graph,
        'system_one_config': system_one_config,
        'system_two_config': system_two_config,
    }

    net = CompLCGNS(**comp_net_config)
    params = net.init(init_rng, init_graph, init_control, net_rng)
    tx = optax.adam(1e-3)

    state = TrainState.create(
        apply_fn=net.apply,
        params=params,
        tx=tx,
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
        
        def forward_pass(graph, inputs):  
            control, t = inputs         
            graph = state.apply_fn(state.params, graph, control, t, jax.random.key(config.seed))
            pred_data = get_pred_data(graph)
            graph = graph._replace(globals=None)
            return graph, pred_data

        _, pred_data = jax.lax.scan(forward_pass, graph, (controls, t0_idxs * net.dt))
        
        losses = [
            jnp.sum(optax.l2_loss(predictions=pred_data[i], targets=exp_data[i])) for i in range(len(exp_data))
        ]
        eval_metrics = [EvalMetrics.single_from_model_output(loss=loss) for loss in losses]
        return ts, np.array(pred_data), np.array(exp_data), eval_metrics
    
    print("Evaluating composition")
    error_sums = [0] * eval_gb._num_states
    for i in range(eval_gb._num_trajectories): # eval_gb._num_trajectories
        ts, pred_data, exp_data, eval_metrics = rollout(state, traj_idx=i)
        for j in range(len(error_sums)):
            error_sums[j] += eval_metrics[j].compute()['loss']
        plot_evaluation_curves(
            ts, 
            pred_data, 
            exp_data, 
            {'name': 'CompLCCircuits'},
            plot_dir=plot_dir,
            prefix=f'comp_eval_traj_{i}')
    rollout_mean_error = np.array(error_sums) / (eval_gb._num_trajectories * eval_gb._num_timesteps)
    print('Rollout error:', rollout_mean_error)
    metrics = {
        'net_name_one': config.net_one_name,
        'net_name_two': config.net_two_name,
        'dir_one': config.paths.ckpt_one_dir,
        'dir_two': config.paths.ckpt_two_dir,
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