import jraph
import jax.numpy as jnp
import networkx as nx
from environments.double_spring_mass import DoubleMassSpring
from utils.custom_types import GraphLabels

def load_data(data) -> GraphLabels:
    """
        TODO: Load data to custom GraphLabels dictionary
    """
    raise NotImplementedError

def build_double_spring_mass_graph(data, t=0, traj_idx=0) -> jraph.GraphsTuple:
    """
        Convert double spring mass environment to a jraph.GraphsTuple
        where V are the masses and there is an edge e between any two connected masses

        Node features:
        Edge features:

        :param data: double mass spring data generated from environments/double_spring_mass.py
        :param t: time (used to index ground-truth position and momentums from trajectory)
        :param traj_idx: specify which trajectory to use
    """
    config = data['config']
    state = data['state_trajectories']

    mass = jnp.array([100, config['m1'], config['m2']]).T
    spring_constant = [config['k1'], config['k2']]
    damping_constant = [config['b1'], config['b2']]

    # position: qs[t] gives position at time t
    qs = jnp.vstack((jnp.zeros(shape=jnp.shape(state[traj_idx,:,0])),
                     state[traj_idx,:,0], 
                     state[traj_idx,:,2])).T # q_wall, q1, q2
    
    # relative positions: dqs[t] gives relative position at time t
    dqs = jnp.column_stack((qs[:,1] - qs[:,0], qs[:,2] - qs[:,1]))
    
    assert(qs[0,1] - qs[0,0] == dqs[0,0])

    # conjugate momentums
    ps = jnp.vstack((jnp.zeros(shape=jnp.shape(state[traj_idx,:,0])), 
          state[traj_idx,:,1], 
          state[traj_idx,:,3])).T # p_wall, p1, p2
    
    # velocities
    vs = ps / mass.reshape(1,-1)
    
    nodes = jnp.column_stack((mass, vs[t])) # shape = (n_node, n_node_feats)
    edges = (dqs[t]).reshape(-1,1)          # shape = (n_edge, n_edge_feats)
    senders = jnp.array([0,1])
    receivers = jnp.array([1,2])

    n_node = jnp.array([len(mass)])
    n_edge = jnp.array([len(spring_constant)])

    global_context = jnp.array([t]).reshape(-1,1) # shape = (n_global_feats, 1) TODO: add ICs

    graph = jraph.GraphsTuple(
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