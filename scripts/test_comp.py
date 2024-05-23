import os
import json
import ml_collections
import optax
from clu import metric_writers
from clu import checkpoint
from scripts.graph_builder import *
from scripts.models import *
from utils.gnn_utils import *
from utils.data_utils import *
from utils.jax_utils import *
from config import *

def test_comp(config: ml_collections.ConfigDict):
    """
        Test merging nodes idea
        1. Given initial condition of composed system, get corresponding graphs of subsystems (with same initial conditions as composite system)
        2. Predict next acceleration of subsystem graphs from subsystem GNSs
        3. Integrate acceleration to get next state of composite system. Add acceleration at merged nodes, or else the acceleration remain the same.
    """
    training_params = config.training_params
    net_params_one = config.net_params_one
    net_params_two = config.net_params_two
    net_params_c = config.net_params_c
    paths = config.paths

    prediction = net_params_one.prediction
    undirected_edges = net_params_one.add_undirected_edges
    self_loops = net_params_one.add_self_loops
    vel_history = net_params_one.vel_history
    control_history = net_params_one.control_history
    num_mp_steps = net_params_one.num_mp_steps

    def create_net(net_params):
        if training_params.net_name == 'GNS':
            return GraphNetworkSimulator(**net_params)
            
    log_dir = os.path.join(paths.dir, 'log')
    checkpoint_dir = os.path.join(paths.dir, 'checkpoint')
    plot_dir = os.path.join(paths.dir, 'eval_plots')

    rng = jax.random.key(training_params.seed)
    rng, init_rng, net_rng = jax.random.split(rng, 3)

    # Create writer for logs
    writer = metric_writers.create_default_writer(logdir=log_dir)

    tx = optax.adam(**config.optimizer_params)

    gb_one = MSDGraphBuilder(paths.evaluation_data_path_one, 
                             undirected_edges, 
                             self_loops, 
                             prediction, 
                             vel_history,
                             control_history)
    
    gb_two = MSDGraphBuilder(paths.evaluation_data_path_two, 
                             undirected_edges, 
                             self_loops, 
                             prediction, 
                             vel_history,
                             control_history)
    
    gb_c = MSDGraphBuilder(paths.evaluation_data_path_comp,
                           undirected_edges,
                           self_loops,
                           prediction,
                           vel_history,
                           control_history)
    
    assert (gb_one._dt == gb_two._dt == gb_c._dt), \
        'dt should be the same for all GNS'
    dt = gb_one._dt
    
    # Initialize first GNS
    net_params_one.norm_stats = gb_one._norm_stats
    net_one = create_net(net_params_one)
    net_one.training = False

    init_graph_one = gb_one.get_graph(traj_idx=0, t=vel_history+1)
    init_control_one = gb_one._control[0, vel_history+1]
    params_one = net_one.init(init_rng, 
                              init_graph_one,
                              init_control_one, 
                              net_rng)
    tx_one = optax.adam(**config.optimizer_params)
    # batched_apply_one = jax.vmap(net_one.apply, in_axes=(None,0,0,None)) # TODO: remove batch
    state_one = TrainState.create(
        apply_fn=net_one.apply,
        params=params_one,
        tx=tx_one,
    )

    # Load first GNS
    checkpoint_dir_one = os.path.join(checkpoint_dir_one, 'best_model')
    ckpt_one = checkpoint.Checkpoint(checkpoint_dir_one)
    state_one = ckpt_one.restore_or_initialize(state_one)

    # Initialize second GNS
    net_params_two.norm_stats = gb_two._norm_stats
    net_two = create_net(net_params_two)
    net_two.training = False

    init_graph_two = gb_two.get_graph(traj_idx=0, t=vel_history+1)
    init_control_two = gb_two._control[0, vel_history+1]
    params_two = net_two.init(init_rng, 
                              init_graph_two,
                              init_control_two, 
                              net_rng)
    tx_two = optax.adam(**config.optimizer_params)
    state_two = TrainState.create(
        apply_fn=net_two.apply,
        params=params_two,
        tx=tx_two,
    )

    # Load second GNS
    checkpoint_dir_two = os.path.join(checkpoint_dir_two, 'best_model')
    ckpt_two = checkpoint.Checkpoint(checkpoint_dir_two)
    state_two = ckpt_two.restore_or_initialize(state_two)
    
    
    def rollout(state: TrainState, traj_idx: int = 0, t0: int = 0):
        tf_idxs = (t0 + jnp.arange(training_params.rollout_timesteps // num_mp_steps)) * num_mp_steps
        t0 = round(vel_history /  num_mp_steps) * num_mp_steps
        tf_idxs = jnp.unique(tf_idxs.clip(min=t0 + num_mp_steps, max=gb_c._num_timesteps))
        ts = tf_idxs * dt # 
        # Get ground truth control, qs, accs
        controls = gb_c._control[traj_idx, tf_idxs]
        exp_qs_buffer = gb_c._qs[traj_idx, tf_idxs]
        exp_as_buffer = gb_c._accs[traj_idx, tf_idxs]
        # Get initial composed graph
        graph = gb_c.get_graph(traj_idx, t0)
        graph_one_0, graph_two_0 = create_graphs(graph) # TODO: how to get previous graphs?

        def forward_pass(graph, aux_data):
            control = aux_data
            graph_one, graph_two, state = state.apply_fn(
                state.params, graph, jnp.array([control]), jax.random.key(0)
            )
            pred_qs = state[::2]

            pred_accs = graph.nodes[:, :,-1]
            # remove acceleration  
            
            return (graph_one, graph_two), (pred_qs.squeeze(), pred_accs.squeeze())

        # start = default_timer()
        (final_graph_one, final_graph_two), pred_data = jax.lax.scan(forward_pass, (graph_one_0, graph_two_0), controls)
        # end = default_timer()
        # jax.debug.print('Inference time {} [sec] for {} passes', end - start, len(ts))

        eval_pos_loss = optax.l2_loss(predictions=pred_data[0], targets=exp_qs_buffer).mean()

        if prediction == 'acceleration':
            aux_data = (gb_c._m[traj_idx], gb_c._k[traj_idx], gb_c._b[traj_idx])
            return ts, np.array(pred_data), np.array((exp_qs_buffer, exp_as_buffer)), aux_data, EvalMetrics.single_from_model_output(loss=eval_pos_loss)
        

    print(f"Number of parameters {num_parameters(params)}")
    rollout_error_sum = 0
    for i in range(eval_gb._num_trajectories):
        ts, pred_data, exp_data, aux_data, eval_metrics = rollout(state, traj_idx=i)
        writer.write_scalars(i, add_prefix_to_keys(eval_metrics.compute(), f'eval {paths["evaluation_data_path"]}'))
        rollout_error_sum += eval_metrics.compute()['loss']
        plot_evaluation_curves(ts, pred_data, exp_data, aux_data, plot_dir=plot_dir, prefix=f'eval_traj_{i}')

    print('Mean rollout error: ', rollout_error_sum / eval_gb._num_trajectories)

    # Save evaluation metrics to json
    eval_metrics = {
        'mean_rollout_error': (rollout_error_sum / eval_gb._num_trajectories).tolist(),
        'evaluation_data_path': paths.evaluation_data_path,
    }
    eval_metrics_file = os.path.join(plot_dir, 'eval_metrics.js')
    with open(eval_metrics_file, "w") as outfile:
        json.dump(eval_metrics, outfile)
