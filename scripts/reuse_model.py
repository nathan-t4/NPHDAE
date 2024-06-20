"""
    Reuse GNN model trained on one circuit for another circuit.
"""

import os
import jax
import optax
from flax.training.train_state import TrainState
import orbax.checkpoint as ocp

from configs.reuse_model import get_comp_gnn_config
from utils.train_utils import *
from utils.gnn_utils import *

def test_composition(config):
    training_params_1 = config.training_params_1
    net_params_1 = config.net_params_1
    paths = config.paths

    if paths.dir == None:
        config.paths.dir = os.path.join(os.curdir, f'results/ReuseModel/{config.trial_name}')
        paths.dir = config.paths.dir

    plot_dir = os.path.join(paths.dir, 'eval_plots')

    rng = jax.random.key(config.seed)
    rng, init_rng, net_rng = jax.random.split(rng, 3)

    net = create_net(config.system_name, training_params_1, net_params_1)

    net.training = False

    graph_builder = create_graph_builder(config.eval_system_name)
    eval_gb = graph_builder(paths.evaluation_data_path)

    if config.eval_system_name == 'LC1':
        net.edge_idxs = np.array([[0,2]])
        net.J = jnp.array([[0, 1, 0],
                        [-1, 0, 1],
                        [0, -1, 0]])
        net.g = jnp.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])
    elif config.eval_system_name == 'LC2':
        net.edge_idxs = np.array([[0]])
        net.J = jnp.array([[0, 1],
                          [-1, 0]])
        net.g = jnp.array([[0, 0],
                          [0, 0]])
    elif config.eval_system_name == 'CoupledLC':
        net.edge_idxs = np.array([[0,2,3]])
        net.J = jnp.array([[0, 1, 0, 0, 0],
                           [-1, 0, 1, 0, 0],
                           [0, -1, 0, 0, -1],
                           [0, 0, 0, 0, 1],
                           [0, 0, 1, -1, 0]])
        net.g = jnp.zeros((5,5))

    net.graph_from_state = eval_gb.get_graph_from_state
    init_control = eval_gb._control[0,0]
    init_graph = eval_gb.get_graph(0, 0)
    
    params = net.init(init_rng, init_graph, init_control, net_rng)

    tx = optax.adam(**config.optimizer_params_1)

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

    time_offset = 1

    def rollout(state, traj_idx, ti = 0):
        tf_idxs = (ti + jnp.arange(1, (config.rollout_timesteps + 1)))
        tf_idxs = jnp.unique(tf_idxs.clip(min=ti + time_offset, max=eval_gb._num_timesteps))
        t0_idxs = tf_idxs - time_offset
        ts = tf_idxs * net.dt
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
    
    print("Evaluating composition")
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
            plot_dir=os.path.join(plot_dir, f'traj_{i}'),
            prefix=f'comp_eval_traj_{i}')
    rollout_mean_error = np.array(error_sums) / eval_gb._num_trajectories

    print(f'State error {rollout_mean_error}')
    metrics = {'rollout_mean_error': rollout_mean_error.tolist()}
    with open(os.path.join(paths.dir, 'metrics.js'), "w") as outfile:
        json.dump(metrics, outfile, indent=4)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default=None)
    args = parser.parse_args()
    cfg = get_comp_gnn_config(args)
    test_composition(cfg)