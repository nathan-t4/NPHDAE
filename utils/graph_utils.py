import flax
import jax
import jraph
import numpy as np
import jax.numpy as jnp
# from scipy.optimize._numdiff import approx_derivative

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

def incidence_matrices_from_graph(graph, edge_types=None):
    ''' Return the incidence matrices of the given graph: (AC, AR, AL, AV, AI) '''
    AC = []
    AR = []
    AL = []
    AV = []
    AI = []
    A = [AC, AR, AL, AV, AI]

    for i in range(len(graph.edges)):
        label = edge_types[i]
        sender_idx = graph.senders[i]
        receiver_idx = graph.receivers[i]
        Ai = np.zeros((len(graph.nodes)))
        Ai[sender_idx] = -1
        Ai[receiver_idx] = 1
        A[label].append(Ai)

    A = [jnp.array(a).T if len(a) > 0 else None for a in A]
    AC, AR, AL, AV, AI = A
    splits = np.array([len(AC.T), # q
                       len(AC.T) + len(AL.T), # q, phi
                       len(AC.T) + len(AL.T) + len(AC), # q, phi, e
                      ]) 
    
    return A, splits

def jac(t, y, yp, F, f=None):
    n = len(y)
    z = jnp.concatenate((y, yp))

    def fun_composite(t, z):
        y, yp = z[:n], z[n:]
        return F(t, y, yp)
    
    J = jax.jacfwd(lambda z: fun_composite(t, z))(z)
    J = J.reshape((n, 2 * n))
    Jy, Jyp = J[:, :n], J[:, n:]
    return Jy, Jyp