import jraph
import jax.numpy as jnp

def explicit_unbatch_graph(graph, system_configs):
    num_nodes = [cfg['num_nodes'] for cfg in system_configs]
    state_dim = [cfg['state_dim'] for cfg in system_configs]
    node_fis = jnp.cumsum(num_nodes)
    node_iis = jnp.r_[jnp.array([0]), node_fis[:,-1]]
    state_fis = jnp.cumsum(state_dim)
    state_iis = jnp.r_[jnp.array([0]), state_fis[:,-1]]

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