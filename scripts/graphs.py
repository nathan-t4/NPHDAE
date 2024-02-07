import jax
import jraph
import numpy as np
import jax.numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt

import utils.graph_utils as graph_utils

def build_graph(dataset_path: str, 
                key, # PRNGKey
                batch_size: int = 1,
                add_undirected_edges: bool = False,
                add_self_loops: bool = False,
                render: bool = False) -> jraph.GraphsTuple:
    """
        Returns graph generated using the dataset config

        :param path: path to dataset
        :param key: jax PRNG key
        :param dataset_type: sample from training or validation dataset
        :param batch_size: num of graphs to return
        :param add_undirected_edges: Add undirected edges to graph
        :param add_self_loops: Add self edges to graph
        :param render: whether to render graph using networkx
    """
    # Load data from path
    data = np.load(dataset_path, allow_pickle=True)
    # Shuffle data (axis 0 = trajectories)
    data['state_trajectories'] = jax.random.permutation(key, data['state_trajectories'], axis=0)
    # Get number of trajectories and timesteps from dataset
    num_trajs, num_timesteps, _ = np.shape(data['state_trajectories'])
    # Sample random times
    rnd_times = np.random.randint(low=0, high=num_timesteps-1, size=batch_size)
    rnd_traj_idx = np.random.randint(low=0, high=num_trajs, size=batch_size)

    # Training/validation split based on traj_idx
    # DATA_SPLIT_PERCENTAGE = 0.8
    # if dataset_type == 'training':
    #     rnd_traj_idx = np.random.randint(low=0, high=DATA_SPLIT_PERCENTAGE * num_trajs, size=batch_size)
    # elif dataset_type == 'validation':
    #     rnd_traj_idx = np.random.randint(low=DATA_SPLIT_PERCENTAGE * num_trajs, high=num_trajs, size=batch_size)
    # else:
    #     rnd_traj_idx = np.zeros(shape=batch_size, dtype=np.int32)
    
    # TODO: efficiently batch - jraph batch, padding, masks, ...
    graphs = []
    for i in range(batch_size):
        graphs.append(build_double_spring_mass_graph(data=data,
                                                     t=rnd_times[i], 
                                                     traj_idx=rnd_traj_idx[i]))

    # Merge all graphs into one using jraph.batch (TODO)
    # graphs = jraph.batch(graphs)

    if add_undirected_edges:
        for i in range(len(graphs)):
            graphs[i] = graph_utils.add_undirected_edges(graphs[i])

    if add_self_loops:
        for i in range(len(graphs)):
            graphs[i] = graph_utils.add_self_loops(graphs[i])

    if render:
        draw_jraph_graph_structure(graphs[0])
        plt.show()

    return graphs


def build_double_spring_mass_graph(data, 
                                   t: int = 0, 
                                   traj_idx: int = 0) -> jraph.GraphsTuple:
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
    spring_constant = jnp.array([config['k1'], config['k2']])
    damping_constant = jnp.array([config['b1'], config['b2']])


    # position: qs[t] gives position at time t
    qs = jnp.vstack((jnp.zeros(shape=jnp.shape(state[traj_idx,:,0])),
                     state[traj_idx,:,0], 
                     state[traj_idx,:,2]),
                     dtype=jnp.float32).T # q_wall, q1, q2
    
    # relative positions: dqs[t] gives relative position at time t
    dqs = jnp.column_stack((qs[:,1] - qs[:,0], qs[:,2] - qs[:,1])).astype(dtype=jnp.float32)
    
    assert(qs[0,1] - qs[0,0] == dqs[0,0])

    # conjugate momentums
    ps = jnp.vstack((jnp.zeros(shape=jnp.shape(state[traj_idx,:,0])), 
          state[traj_idx,:,1], 
          state[traj_idx,:,3])).T # p_wall, p1, p2
    
    # velocities
    vs = ps / mass.reshape(1,-1)

    n_node = jnp.array([len(mass)])         # num nodes
    n_edge = jnp.array([jnp.shape(dqs)[1]]) # num edges

    # Test embeddings
    # nodes = vs[t].reshape(n_node.item(), -1)             # shape = (n_node, n_node_feats)
    # edges = (dqs[t]).reshape(n_edge.item(), -1)          # shape = (n_edge, n_edge_feats)

    # Following embeddings from "Learning to Simulate Complex Physics with Graph Networks"
    nodes = jnp.concatenate((qs[t], vs[t])).reshape(n_node.item(), -1)
    edges = dqs[t].reshape((n_edge.item(), -1))
    # edges = jnp.concatenate((dqs[t], spring_constant, damping_constant)).reshape(n_edge.item(), -1)
    
    senders = jnp.array([0,1])
    receivers = jnp.array([1,2])

    # global context is [t, masses, ICs]. shape = (n_global_feats, 1) 
    global_context = jnp.concatenate((jnp.array([t]), mass, qs[0]), dtype=jnp.int32).reshape(-1,1)

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