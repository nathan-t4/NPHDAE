import jraph
import jax.numpy as jnp

def explicit_unbatch_graph(graph, system_configs):
    num_nodes = [cfg['num_nodes'] for cfg in system_configs]
    state_dim = [cfg['state_dim'] for cfg in system_configs]
    node_fis = jnp.cumsum(num_nodes)
    node_iis = jnp.r_[jnp.array([0]), node_fis[:-1]]
    state_fis = jnp.cumsum(state_dim)
    state_iis = jnp.r_[jnp.array([0]), state_fis[:-1]]

    nodes = [graph.nodes[ni, nf] for ni, nf in zip(node_iis, node_fis)]
    edges = [graph.edges[ei, ef] for ei, ef in zip(state_iis, state_fis)]
    receivers = [graph.receivers[ei, ef] for ei, ef in zip(state_iis, state_fis)]
    senders = [graph.senders[ei, ef] for ei, ef in zip(state_iis, state_fis)]
    graphs = []
    for i in range(len(system_configs)):
        graph_i = jraph.GraphsTuple(
            nodes=nodes[i],
            edges=edges[i],
            receivers=receivers[i],
            senders=senders[i],
            n_node=jnp.array([len(nodes[i])]),
            n_edge=jnp.array([len(edges[i])]),
        )
        graphs.append(graph_i)

    return graphs

def explicit_unbatch_control(control, system_configs):
    num_subsystems = len(system_configs)
    num_cur_sources = [cfg['num_cur_sources'] for cfg in system_configs]
    num_volt_sources = [cfg['num_volt_sources'] for cfg in system_configs]
    ext_curs = [control[sum(num_cur_sources[:i]) : sum(num_cur_sources[:i+1])] for i in range(num_subsystems-1)]
    ext_volts = [
        control[
            sum(num_cur_sources)+sum(num_volt_sources[:i]) : 
            sum(num_cur_sources)+sum(num_volt_sources[:i+1])
            ]    
        for i in range(num_subsystems-1)
        ]
    
    controls = [jnp.concatenate((i, v)) for i, v in zip(ext_curs, ext_volts)]

    return controls