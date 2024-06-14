"""
    Test composition of circuits 1 and 2 to get coupled lc circuit

    How to compose:
    - Concatenate states of circuits 1 and 2 to get state of coupled lc circuit
    - modify circuit 1 and 2 graphs to satisfy Kirchhoff's laws that arise from composition
    - Predict Hamiltonian of circuits 1 and 2 from respective GNS
        - make sure nodes of circuits are correct, then edges of GNS predict energy.
    - Add Hamiltonian and use PH dynamics to get next state (decoder)

    Pipeline
    - load trained models GNS1 and GNS2 for circuits 1 and 2
    - decompose an initial coupled lc graph into graph1 and graph2
    - feed into CompGNS to get next_graph1 and next_graph2
    - repeat for # of rollout timesteps
    - compare results from CompGNS vs expected results

    # TODO:
    - why isn't it working?
    - GNSs predict energies associated with edges of graphs 1 and 2
"""

import os
import jax
import optax
from time import strftime
from clu import checkpoint
from flax.training.train_state import TrainState
import orbax.checkpoint as ocp

from scripts.graph_nets import CompLCGNS
from configs.comp_circuits import get_comp_gnn_config
from utils.train_utils import *
from utils.gnn_utils import *

def decompose_coupled_lc_graph(lc_graph):
    """
    nodes = jnp.array([[0], [V2], [V3]]) # 1 2 3
    edges = jnp.array([[Q1], [Phi1], [Q3]) # e1 e2 e3
    senders = jnp.array([0, 1, 0])
    receivers = jnp.array([1, 2, 2])

    nodes = jnp.array([[V_ext], [Vc], [0]]) # 4 5 6
    edges = jnp.array([[Q2], [Phi2]]) # e5 e4
    senders = jnp.array([2, 1])
    receivers = jnp.array([1, 0])

    nodes = jnp.array([[0], [V2], [V3], [V4]]) # 1 2 3 4
    edges = jnp.array([[Q1], [Phi1], [Q3], [Q2], [Phi2]]) # e1 e2 e3 e4 e5
    senders = jnp.array([0, 1, 0, 3, 0])
    receivers = jnp.array([1, 2, 2, 2, 3])
    """
    n1_idx = jnp.array([0,1,2]) # [[0], [V2], [V3]]
    n2_idx = jnp.array([2,3,0]) # [[V3], [V4], 0]
    e1_idx = jnp.array([0,1,2]) # [[Q1], [Phi1], [Q3]] 
    e2_idx = jnp.array([3,4]) # [[Q2], [Phi2]] 
    nodes_1 = lc_graph.nodes[n1_idx]
    edges_1 = lc_graph.edges[e1_idx]
    n_node_1 = jnp.array([len(nodes_1)])
    n_edge_1 = jnp.array([len(edges_1)])
    receivers_1 = lc_graph.receivers[e1_idx]
    senders_1 = lc_graph.senders[e1_idx]

    graph_1 = jraph.GraphsTuple(nodes=nodes_1,
                                edges=edges_1,
                                globals=None,
                                senders=senders_1,
                                receivers=receivers_1,
                                n_node=n_node_1,
                                n_edge=n_edge_1)

    nodes_2 = lc_graph.nodes[n2_idx]
    edges_2 = lc_graph.edges[e2_idx]
    n_node_2 = jnp.array([len(nodes_2)])
    n_edge_2 = jnp.array([len(edges_2)])
    senders_2 = jnp.array([2, 1])
    receivers_2 = jnp.array([1, 0])

    graph_2 = jraph.GraphsTuple(nodes=nodes_2,
                                edges=edges_2,
                                globals=None,
                                senders=senders_2,
                                receivers=receivers_2,
                                n_node=n_node_2,
                                n_edge=n_edge_2)

    return graph_1, graph_2

def test_composition(config):
    training_params_1 = config.training_params_1
    net_params_1 = config.net_params_1
    training_params_2 = config.training_params_2
    net_params_2 = config.net_params_2
    paths = config.paths

    if paths.dir == None:
        config.paths.dir = os.path.join(os.curdir, f'results/CompGNN/{config.trial_name}')
        paths.dir = config.paths.dir

    plot_dir = os.path.join(paths.dir, 'eval_plots')

    rng = jax.random.key(config.seed)
    rng, init_rng, net_rng = jax.random.split(rng, 3)

    graph_builder = create_graph_builder('CoupledLC')

    eval_gb = graph_builder(paths.coupled_lc_data_path)

    net_one = create_net('LC1', training_params_1, net_params_1)
    
    net_two = create_net('LC2', training_params_2, net_params_2)

    net_one.training = False
    net_two.training = False

    init_control = eval_gb._control[0,0]

    init_graph_one, init_graph_two = decompose_coupled_lc_graph(eval_gb.get_graph(traj_idx=0, t=0))

    params_one = net_one.init(init_rng, init_graph_one, init_control[:3], net_rng)

    params_two = net_two.init(init_rng, init_graph_two, init_control[3:], net_rng)

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

    state_one = ckpt_mngr_one.restore(paths.ckpt_one_step, args=ocp.args.StandardRestore(state_one))
    state_two = ckpt_mngr_two.restore(paths.ckpt_two_step, args=ocp.args.StandardRestore(state_two))

    assert net_one.dt == net_two.dt

    dt = net_one.dt

    # Initialize composite GNS
    net = CompLCGNS('euler', eval_gb._dt, state_one, state_two)
    params = net.init(init_rng, init_graph_one, init_graph_two, net_rng)
    tx = optax.adam(1e-3)

    state = TrainState.create(
        apply_fn=net.apply,
        params=params,
        tx=tx,
    )

    time_offset = 1

    def rollout(state, traj_idx, ti = 0):
        tf_idxs = (ti + jnp.arange(1, (config.rollout_timesteps + 1)))
        tf_idxs = jnp.unique(tf_idxs.clip(min=ti + time_offset, max=eval_gb._num_timesteps))
        t0_idxs = tf_idxs - ti
        ts = tf_idxs * net.dt
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
            pred_Q1 = (next_graph_one.edges[0]).squeeze()
            pred_Phi1 = (next_graph_one.edges[1]).squeeze()
            pred_Q3 = (next_graph_one.edges[2]).squeeze()
            pred_Q2 = (next_graph_two.edges[0]).squeeze()
            pred_Phi2 = (next_graph_two.edges[1]).squeeze()
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
            plot_dir=os.path.join(plot_dir, f'traj_{i}'),
            prefix=f'comp_eval_traj_{i}')
    rollout_mean_error = np.array(error_sums) / eval_gb._num_trajectories

    print(f'State error {rollout_mean_error}')

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default=None)
    args = parser.parse_args()
    cfg = get_comp_gnn_config(args)
    test_composition(cfg)