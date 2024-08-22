import os
import jax
import optax
from flax.training.train_state import TrainState
import orbax.checkpoint as ocp

from scripts.model_instances.comp_nets import CompPHGNS
from helpers.graph_builder_factory import gb_factory
from helpers.config_factory import config_factory
from utils.train_utils import *
from utils.gnn_utils import *
from utils.comp_utils import *

def compose(config):
    subsystem_names = config.subsystem_names
    learned_subsystem = config.learned_subsystem
    num_subsystems = len(subsystem_names)
    net_params = config.net_params
    optimizer_params = config.optimizer_params
    paths = config.paths
    ckpt_dirs = paths.ckpt_dirs
    ckpt_steps = paths.ckpt_steps
    Alambda = config.Alambda
    comp_As = config.composite_incidence_matrices
    comp_incidence_matrices = (
        comp_As.AC, comp_As.AR, comp_As.AL, comp_As.AV, comp_As.AI
    )

    if paths.dir == None:
        config.paths.dir = os.path.join(os.curdir, f'results/CompGNN/{config.trial_name}')
        paths.dir = config.paths.dir

    plot_dir = os.path.join(paths.dir, 'eval_plots')

    rng = jax.random.key(config.seed)
    rng, init_rng, net_rng = jax.random.split(rng, 3)

    graph_builder = gb_factory('') # Get TestGraphBuilder

    eval_gb = graph_builder(paths.comp_data_path, *comp_incidence_matrices)

    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)  

    gbs = []
    graph_to_state = []
    state_to_graph = []
    alg_vars_from_graph = []
    incidence_matrices = []
    system_configs = []
    ACs = []; ARs = []; ALs = []; AVs = []; AIs = []

    for i in range(num_subsystems):
        AC = config.incidence_matrices[i].AC
        AR = config.incidence_matrices[i].AR
        AL = config.incidence_matrices[i].AL
        AV = config.incidence_matrices[i].AV
        AI = config.incidence_matrices[i].AI
        
        incidence_matrices.append((AC, AR, AL, AV, AI))

        ACs.append(AC)
        ARs.append(AR)
        ALs.append(AL)
        AVs.append(AV)
        AIs.append(AI)

        if learned_subsystem[i]:
            gb = gb_factory(subsystem_names[i])(
                paths.training_data_paths[i], *incidence_matrices[i]
                )
            gbs.append(gb)

            g_to_s = gb.graph_to_state
            s_to_g = gb.state_to_graph
            alg_from_g = gb.get_alg_vars_from_graph
            
            graph_to_state.append(g_to_s)
            state_to_graph.append(s_to_g)
            alg_vars_from_graph.append(alg_from_g)
            net_params[i].training = False 
            net_params[i].graph_to_state = g_to_s
            net_params[i].state_to_graph = s_to_g
            net_params[i].alg_vars_from_graph = alg_from_g
            net_params[i].edge_idxs = gb.edge_idxs
            net_params[i].node_idxs = gb.node_idxs
            net_params[i].include_idxs = gb.include_idxs
            system_config = get_system_config(*incidence_matrices[i])
            system_config['is_k'] = False

        else:
            gb = gb_factory(subsystem_names[i])(
               paths.training_data_paths[i], *incidence_matrices[i]
                )
            gbs.append(gb)

            g_to_s = gb.graph_to_state
            s_to_g = gb.state_to_graph
            alg_from_g = gb.get_alg_vars_from_graph
            
            graph_to_state.append(g_to_s)
            state_to_graph.append(s_to_g)
            alg_vars_from_graph.append(alg_from_g)

            # TODO: move somewhere...
            # This is Alambda for subsystem k (transmission line)
            Alambda_k = Alambda[3:6] # (num_nodes_1) : (num_nodes_1+num_nodes_2)
            # Alambda_k = jnp.concatenate((jnp.zeros((1,2)), Alambda_k)) # add ground node
            system_config = get_system_k_config(*incidence_matrices[i], Alambda_k)
            system_config['is_k'] = True
        
        system_configs.append(system_config)

    init_t = 0.0
    init_graph = eval_gb.get_graph(traj_idx=0, t=0)
    init_graphs = explicit_unbatch_graph(init_graph, Alambda, system_configs)
    # init_graph = jraph.batch(init_graphs)
    # init_graph = explicit_batch_graphs(
    #     init_graphs, Alambda, system_configs, init_graph.senders, init_graph.receivers
    #     )

    init_control = eval_gb._control[0,0]
    init_controls = explicit_unbatch_control(init_control, system_configs)

    nets = []
    params = []
    txs = []
    train_states = []
    # Restore GNNs
    for i in range(num_subsystems):
        if learned_subsystem[i]:
            net_params[i].system_config = system_configs[i]
            net = create_net(net_params[i])
            nets.append(net)

            param = net.init(init_rng, init_graphs[i], init_controls[i], init_t, net_rng)
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
        else:
            # For known systems append None
            nets.append(None)
            params.append(None)
            txs.append(None)
            train_states.append(None)

    # TODO: ignore for now because net[k] = None
    # assert all([net.dt == nets[0].dt for net in nets])
    # assert all([net.T == nets[0].T for net in nets])
    dt = nets[0].dt
    T = nets[0].T
    ode_integration_method = 'adam_bashforth'

    # Initialize composite GNS
    comp_sys_config = get_composite_system_config(system_configs, Alambda)

    comp_net_config = {
        'ode_integration_method': ode_integration_method,
        'dt': dt,
        'T': T,
        'train_states': train_states,
        'graph_to_state': graph_to_state,
        'state_to_graph': state_to_graph,
        'alg_vars_from_graph': alg_vars_from_graph,
        'system_configs': system_configs,
        'composite_system_config': comp_sys_config,
        'Alambda': Alambda,
    }

    num_lambs = len(Alambda.T)
    init_lamb = None # jnp.zeros(num_lambs) # None 
    comp_net = CompPHGNS(**comp_net_config)
    comp_params = comp_net.init(
        init_rng, init_graphs, init_control, init_lamb, init_t, net_rng
        )
    comp_tx = optax.adam(1e-3) # The learning rate is irrelevant

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
        graph = explicit_unbatch_graph(graph, Alambda, system_configs)
        # graph = jraph.batch(graph)
        controls = eval_gb.get_control(traj_idx, t0_idxs)
        exp_data = eval_gb.get_exp_data(traj_idx, tf_idxs)
        get_pred_data = eval_gb.get_pred_data
        # get_batch_pred_data = eval_gb.get_batch_pred_data
        
        def forward_pass(carry, inputs):  
            graph, lamb = carry
            control, t = inputs         
            graphs, next_lamb = state.apply_fn(
                state.params, graph, control, lamb, t, jax.random.key(config.seed)
            )
            # TODO: make sure graphs are correct...
            # TODO: merge all graphs to graph
            graph = explicit_batch_graphs(graphs)
            pred_data = get_pred_data(graphs)
            graphs = [graph._replace(globals=None) for graph in graphs]
            return (graphs, next_lamb), pred_data

        # TODO
        # init_lamb = jnp.zeros(num_lambs)
        init_lamb = None
        init_timesteps = t0_idxs * dt
        _, pred_data = jax.lax.scan(forward_pass, (graph, init_lamb), (controls, init_timesteps))
        
        losses = [
            jnp.sum(optax.l2_loss(predictions=pred_data[i], targets=exp_data[i])) for i in range(len(exp_data))
        ]
        eval_metrics = [EvalMetrics.single_from_model_output(loss=loss) for loss in losses]
        pred_data = np.concatenate(pred_data, axis=1)
        exp_data = np.concatenate(exp_data, axis=1)
        return pred_data, exp_data, eval_metrics
    
    
    print('##################################################')
    subsystem_names = np.array(subsystem_names)
    print(f"Evaluating composition of subsystems {subsystem_names}")
    print(f"The known subsystems are {subsystem_names[[i for i, b in enumerate(learned_subsystem) if b]]}")
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
        'learned_subsystem': learned_subsystem,
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
    cfg = config_factory('CompCircuits', args) # Old one is 'CompCircuitsOld'
    compose(cfg)