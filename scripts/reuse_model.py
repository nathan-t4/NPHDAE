"""
    Reuse GNN model trained on one circuit for another circuit.
"""

import os
import jax
import optax
from flax.training.train_state import TrainState
from flax.core import frozen_dict
import orbax.checkpoint as ocp

from configs.reuse_model import get_reuse_model_config
from helpers.graph_builder_factory import gb_factory
from utils.train_utils import *
from utils.gnn_utils import *

def transfer(config):
    training_params_1 = config.training_params_1
    net_params_1 = config.net_params_1
    paths = config.paths

    if paths.dir == None:
        config.paths.dir = os.path.join(os.curdir, f'results/ReuseModel/{config.trial_name}')
        paths.dir = config.paths.dir

    plot_dir = os.path.join(paths.dir, 'eval_plots')

    rng = jax.random.key(config.seed)
    rng, init_rng, net_rng = jax.random.split(rng, 3)

    net_params_1.training = False

    net = create_net(config.system_name, net_params_1)

    graph_builder = gb_factory(config.eval_system_name)
    eval_gb = graph_builder(paths.evaluation_data_path)
    net.edge_idxs = eval_gb.edge_idxs
    net.node_idxs = eval_gb.node_idxs

    if not training_params_1.learn_matrices:
        J, R, g = get_pH_matrices(config.eval_system_name)
        net.J = J
        net.R = R
        net.g = g

    net.graph_from_state = eval_gb.get_graph_from_state
    init_control = eval_gb._control[0,0]
    init_graph = eval_gb.get_graph(0, 0)
    
    params = net.init(init_rng, init_graph, init_control, net_rng)

    # print(params)

    tx = optax.adam(**config.optimizer_params_1)
    # Need to freeze params
    # tx = optax.multi_transform({'adam': optax.adam(**config.optimizer_params_1), 
    #                             'zero': optax.set_to_zero()}, 
    #                             frozen_dict.freeze({'params':{'update_edge': 'adam', 'update_node': 'adam', 'enc_node': 'adam', 'enc_edge_1': 'adam', 'enc_edge_2': 'adam', 'dec_node': 'adam', 'dec_edge_1': 'adam', 'dec_edge_2': 'adam', 'GNS': 'adam', 'J': 'zero', 'g': 'zero'} }))

    state = TrainState.create(
        apply_fn=net.apply,
        params=params,
        tx=tx,
    )

    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)  
    paths.ckpt_one_dir = os.path.abspath(paths.ckpt_one_dir)  
    ckpt_mngr = ocp.CheckpointManager(
        paths.ckpt_one_dir,
        options=options,
        item_handlers=ocp.StandardCheckpointHandler(),
    )

    state = ckpt_mngr.restore(paths.ckpt_one_step, args=ocp.args.StandardRestore(state))

    T = net.T
    dt = net.dt

    def rollout(state, traj_idx, ti = 0):
        tf_idxs = ti + jnp.arange(1, jnp.floor_divide(config.rollout_timesteps + 1, T))
        tf_idxs = jnp.unique(tf_idxs.clip(min=ti + 1, max=jnp.floor_divide(eval_gb._num_timesteps + 1, T))) * T
        t0_idxs = tf_idxs - T
        ts = tf_idxs * dt
        graphs = eval_gb.get_graph(traj_idx, ti)
        controls = eval_gb.get_control(traj_idx, t0_idxs)
        exp_data = eval_gb.get_exp_data(traj_idx, tf_idxs)
        get_pred_data = eval_gb.get_pred_data
        batch_data = (tf_idxs, controls) # tf_idxs as placeholder
        if config.eval_system_name == 'LC1':
            batch_data = ((eval_gb._V2[traj_idx, tf_idxs], eval_gb._V3[traj_idx, tf_idxs]), controls)
        elif config.eval_system_name == 'CoupledLC':
            batch_data = ((eval_gb._V2[traj_idx, tf_idxs], eval_gb._V3[traj_idx, tf_idxs], eval_gb._V4[traj_idx, tf_idxs]), controls)

        def forward_pass(graphs, data):
            node_features, control = data
            graph = graphs
            next_graph = state.apply_fn(state.params, graph, control, jax.random.key(config.seed))
            pred_data = get_pred_data(next_graph)

            if config.set_nodes:
                if config.eval_system_name == 'LC1':
                    V2, V3 = node_features
                    next_nodes = jnp.array([[0], [V2], [V3]])
                elif config.eval_system_name == 'CoupledLC':
                    V2, V3, V4 = node_features
                    next_nodes = jnp.array([[0], [V2], [V3], [V4]])
                next_graph = next_graph._replace(nodes=next_nodes, globals=None)

            next_graph = next_graph._replace(globals=None)

            return (next_graph), pred_data
        
        _, pred_data = jax.lax.scan(forward_pass, graphs, batch_data)
        losses = [jnp.sum(optax.l2_loss(pred_data[i], exp_data[i])) for i in range(len(exp_data))]
        eval_metrics = [EvalMetrics.single_from_model_output(loss=loss) for loss in losses]
        
        return ts, np.array(pred_data), np.array(exp_data), eval_metrics
    
    print("Evaluating zero-shot transfer")
    error_sums = [0] * eval_gb._num_states
    for i in range(eval_gb._num_trajectories):
        ts, pred_data, exp_data, eval_metrics = rollout(state, traj_idx=i)
        for j in range(len(error_sums)):
            error_sums[j] += eval_metrics[j].compute()['loss']
        plot_evaluation_curves(
            ts, 
            pred_data, 
            exp_data, 
            {'name': config.eval_system_name},
            plot_dir=plot_dir,
            prefix=f'comp_eval_traj_{i}')
    rollout_mean_error = np.array(error_sums) / (eval_gb._num_trajectories * eval_gb._num_timesteps)

    print(f'State error {rollout_mean_error}')
    print(f'Mean error {rollout_mean_error.mean()}')

    metrics = {
        'rollout_mean_error_states': rollout_mean_error.tolist(),
        'rollout_mean_error': str(rollout_mean_error.mean())
    }
    with open(os.path.join(paths.dir, 'metrics.js'), "w") as outfile:
        json.dump(metrics, outfile, indent=4)

    return metrics['rollout_mean_error']

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default=None)
    args = parser.parse_args()
    cfg = get_reuse_model_config(args)
    transfer(cfg)