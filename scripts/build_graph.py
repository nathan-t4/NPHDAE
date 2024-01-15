import jraph
import jax.numpy as jnp
import networkx as nx
from environments.double_spring_mass import DoubleMassSpring

def build_double_spring_mass_graph(data) -> jraph.GraphsTuple:
    # TODO: add self edges?
    print(data['config'])
    config = data['config']
    state = data['state_trajectories']

    mass = [100, config['m1'], config['m2']]
    spring_constant = [config['k1'], config['k2']]
    damping_constant = [config['b1'], config['b2']]
    position = [state[0,:,0], state[0,:,2]]
    momentum = [state[0,:,1], state[0,:,3]]
    
    nodes = jnp.array(mass).T # this should be shape = (n_nodes, n_node_features)
    edges = jnp.ones(len(spring_constant)).T
    senders = jnp.array([0,1])
    receivers = jnp.array([1,2])

    n_node = jnp.array([len(mass)])
    n_edge = jnp.array([len(spring_constant)])

    global_context = jnp.array([[]]) #

    graph = jraph.GraphsTuple(
        # nodes={
        #     "mass": mass,
        #     "position": position,
        #     "momentum": momentum,
        # },
        # edges={
        #     "spring_constant": spring_constant,
        #     "damping_constant": damping_constant,
        # },
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        globals=global_context,
    )
    return graph

def convert_jraph_to_networkx_graph(jraph_graph: jraph.GraphsTuple) -> nx.Graph:
    nodes, edges, receivers, senders, _, _, _ = jraph_graph
    ### for testing double spring mass damper graph
    # nodes = nodes['mass']
    # edges = edges['spring_constant']
    ###
    nx_graph = nx.DiGraph()
    if nodes is None:
        for n in range(jraph_graph.n_node[0]):
            nx_graph.add_node(n)
    else:
        for n in range(jraph_graph.n_node[0]):
            nx_graph.add_node(n, node_feature=nodes[n])
    if edges is None:
        for e in range(jraph_graph.n_edge[0]):
            nx_graph.add_edge(int(senders[e]), int(receivers[e]))
    else:
        for e in range(jraph_graph.n_edge[0]):
            nx_graph.add_edge(
            int(senders[e]), int(receivers[e]), edge_feature=edges[e])
    return nx_graph


def draw_jraph_graph_structure(jraph_graph: jraph.GraphsTuple) -> None:
    nx_graph = convert_jraph_to_networkx_graph(jraph_graph)
    pos = nx.spring_layout(nx_graph)
    nx.draw(nx_graph, pos=pos, with_labels=True, node_size=500, font_color='yellow')