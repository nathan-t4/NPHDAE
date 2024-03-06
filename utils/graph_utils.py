import flax
import jraph
import jax.numpy as jnp

def add_edges(graph, undirected_edges, self_loops):
    if undirected_edges:
        graph = add_undirected_edges(graph)
    if self_loops:
        graph = add_self_loops(graph)
    return graph

def add_undirected_edges(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    new_senders = jnp.concatenate((graph.senders, graph.receivers), axis=0)
    new_receivers = jnp.concatenate((graph.receivers, graph.senders), axis=0)
    edges = jnp.concatenate((graph.edges, graph.edges), axis=0)
    n_edge = jnp.array([len(edges)])
    
    return graph._replace(senders=new_senders, 
                          receivers=new_receivers, 
                          edges=edges, 
                          n_edge=n_edge)
    
def add_self_loops(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    n_node = len(graph.nodes)
    edge_feature_dim = jnp.shape(graph.edges)[1]
    new_senders = jnp.concatenate((graph.senders, jnp.arange(n_node)), axis=0)
    new_receivers = jnp.concatenate((graph.receivers, jnp.arange(n_node)), axis=0)
    edges = jnp.concatenate((graph.edges, jnp.zeros((n_node, edge_feature_dim))), axis=0)
    n_edge = jnp.array([len(edges)])

    return graph._replace(senders=new_senders,
                          receivers=new_receivers, 
                          edges=edges, 
                          n_edge=n_edge)   

def check_dictionary(dictionary, condition):
    # Check if condition holds true for all values of the dictionary
    dictionary = flax.core.unfreeze(dict)
    flat_grads = {
        '/'.join(k): v
        for k, v in flax.traverse_util.flatten_dict(dictionary).items()
    }
    cond = True
    for array in flat_grads.values():
        cond = cond and (jnp.all(condition(array)))

    return cond