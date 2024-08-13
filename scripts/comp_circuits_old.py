"""
    Test composition of circuits 1 and 2 to get coupled lc circuit

    How to compose:
    - Concatenate states of circuits 1 and 2 to get state of coupled lc circuit
    - Predict Hamiltonian of circuits 1 and 2 from respective GNS
        - make sure nodes of circuits are correct, then edges of GNS predict energy.
    - Add Hamiltonian and use PH dynamics to get next state (decoder)

    Pipeline
    - load trained models GNS1 and GNS2 for circuits 1 and 2
    - decompose an initial coupled lc graph into graph1 and graph2
    - feed into CompGNS to get next_graph1 and next_graph2
    - repeat for # of rollout timesteps
    - compare results from CompGNS vs expected results
"""

import os
import jax
import optax
from flax.training.train_state import TrainState
import orbax.checkpoint as ocp

from scripts.model_instances.comp_nets import CompLCGNSOld
from configs.comp_circuits import get_comp_gnn_config
from helpers.graph_builder_factory import gb_factory
from utils.train_utils import *
from utils.gnn_utils import *

def decompose_coupled_lc_graph(lc_graph):
    """
    nodes = jnp.array([[0], [V2], [V3]])
    edges = jnp.array([[Q1], [Phi1], [Q3])
    senders = jnp.array([0, 1, 0])
    receivers = jnp.array([1, 2, 2])

    nodes = jnp.array([[0], [V4], [V3]])
    edges = jnp.array([[Q2], [Phi2], [Q3]])
    senders = jnp.array([0, 1, 0])
    receivers = jnp.array([1, 2, 2])

    nodes = jnp.array([[0], [V2], [V3], [V4]])
    edges = jnp.array([[Q1], [Phi1], [Q3], [Q2], [Phi2]])
    senders = jnp.array([0, 1, 0, 3, 0])
    receivers = jnp.array([1, 2, 2, 2, 3])
    """
    Q1, Phi1, Q3, Q2, Phi2 = lc_graph.edges
    zero, V2, V3, V4 = lc_graph.nodes
    nodes_1 = jnp.array([[0], V2, V3]) # [[0], [V2], [V3]]
    edges_1 = jnp.array([Q1, Phi1, Q3]) # [[Q1], [Phi1], [Q3]] 
    n_node_1 = jnp.array([len(nodes_1)])
    n_edge_1 = jnp.array([len(edges_1)])
    senders_1 = jnp.array([0, 1, 0])
    receivers_1 = jnp.array([1, 2, 2])

    graph_1 = jraph.GraphsTuple(nodes=nodes_1,
                                edges=edges_1,
                                globals=None,
                                senders=senders_1,
                                receivers=receivers_1,
                                n_node=n_node_1,
                                n_edge=n_edge_1)

    nodes_2 = jnp.array([[0], V4, V3]) # [0, [V4], [V3]]
    edges_2 = jnp.array([Q2, Phi2, Q3]) # [[Q2], [Phi2], [Q3]] 
    n_node_2 = jnp.array([len(nodes_2)])
    n_edge_2 = jnp.array([len(edges_2)])
    senders_2 = jnp.array([0, 1, 0])
    receivers_2 = jnp.array([1, 2, 2])

    graph_2 = jraph.GraphsTuple(nodes=nodes_2,
                                edges=edges_2,
                                globals=None,
                                senders=senders_2,
                                receivers=receivers_2,
                                n_node=n_node_2,
                                n_edge=n_edge_2)

    return graph_1, graph_2

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
    
    if not config.learn_matrices_one:
        J1, R1, g1 = get_pH_matrices(config.net_one_name)
        net_params_1.J = J1
        net_params_1.R = R1
        net_params_1.g = g1
    if not config.learn_matrices_two:
        J2, R2, g2 = get_pH_matrices(config.net_two_name)
        net_params_2.J = J2
        net_params_2.R = R2
        net_params_2.g = g2

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

    init_control = eval_gb._control[0,0]

    init_graph_one, init_graph_two = decompose_coupled_lc_graph(eval_gb.get_graph(traj_idx=0, t=0))

    params_one = net_one.init(init_rng, init_graph_one, init_control[jnp.array([0,1,2])], net_rng)

    params_two = net_two.init(init_rng, init_graph_two, init_control[jnp.array([3,4,2])], net_rng)

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
    net = CompLCGNSOld(integrator, eval_gb._dt, T, state_one, state_two)
    params = net.init(init_rng, init_graph_one, init_graph_two, net_rng)
    tx = optax.adam(1e-3)

    state = TrainState.create(
        apply_fn=net.apply,
        params=params,
        tx=tx,
    )

    def rollout(state, traj_idx, ti = 0):
        tf_idxs = ti + jnp.arange(1, jnp.floor_divide(config.rollout_timesteps + 1, T))
        tf_idxs = jnp.unique(tf_idxs.clip(min=ti + 1, max=jnp.floor_divide(eval_gb._num_timesteps + 1, T))) * T
        t0_idxs = tf_idxs - T
        ts = tf_idxs * dt
        graphs = decompose_coupled_lc_graph(eval_gb.get_graph(traj_idx, ti)) 

        controls = t0_idxs # used as 'length' for scan loop
        exp_Q1 = eval_gb._Q1[traj_idx, tf_idxs]
        exp_Phi1 = eval_gb._Phi1[traj_idx, tf_idxs]
        exp_Q3 = eval_gb._Q3[traj_idx, tf_idxs]
        exp_Q2 = eval_gb._Q2[traj_idx, tf_idxs]
        exp_Phi2 = eval_gb._Phi2[traj_idx, tf_idxs]
        exp_H = eval_gb._H[traj_idx, tf_idxs]
        exp_data = (exp_Q1, exp_Phi1, exp_Q3, exp_Q2, exp_Phi2, exp_H)

        def forward_pass(graphs, control):
            graph_one, graph_two = graphs
            next_graph_one, next_graph_two = state.apply_fn(state.params, graph_one, graph_two, jax.random.key(0))
            pred_Q1 = (next_graph_one.edges[0,0]).squeeze()
            pred_Phi1 = (next_graph_one.edges[1,0]).squeeze()
            pred_Q3 = (next_graph_one.edges[2,0]).squeeze()
            pred_Q2 = (next_graph_two.edges[0,0]).squeeze()
            pred_Phi2 = (next_graph_two.edges[1,0]).squeeze()
            pred_H = (next_graph_one.globals).squeeze() + (next_graph_two.globals).squeeze()
            
            next_graph_one = next_graph_one._replace(globals=None)
            next_graph_two = next_graph_two._replace(globals=None)

            return (next_graph_one, next_graph_two), (pred_Q1, pred_Phi1, pred_Q3, pred_Q2, pred_Phi2, pred_H)
        
        (_, __), pred_data = jax.lax.scan(forward_pass, graphs, controls)

        losses = [jnp.sum(optax.l2_loss(pred_data[i], exp_data[i])) for i in range(len(exp_data))]
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