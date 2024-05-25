import jraph
import jax.numpy as jnp

def CustomEdgeGraphMapFeatures(embed_edge_fn_1 = None,
                               embed_edge_fn_2 = None,
                               embed_node_fn = None,
                               embed_global_fn = None,
                               edge_idxs = None):
    identity = lambda x : x
    embed_edges_fn_1 = embed_edge_fn_1 if embed_edge_fn_1 else identity
    embed_edges_fn_2 = embed_edge_fn_2 if embed_edge_fn_2 else identity
    embed_nodes_fn = embed_node_fn if embed_node_fn else identity
    embed_globals_fn = embed_global_fn if embed_global_fn else identity
    # edge_idxs = edge_idxs if edge_idxs else None

    def Embed(graph):
        """
        1. differentiate edges_one and edges_two
        2. apply embed_edge_fn_1 on edges_one and embed_edge_fn_2 on edges_two (in place?)
        3. replace graph edges with new edges, nodes with new nodes, and globals with new globals
        4. return graph
        """
        # mask = np.zeros(len(graph.edges))
        # mask[edge_idxs] = 1
        # mask = mask.astype(bool)
        # edges_one = graph.edges[mask]
        # edges_two = graph.edges[~mask]
        new_edges = None

        # for i in range(len(graph.edges)):
        #     if i in edge_idxs:
        #         new_edge = embed_edges_fn_1(graph.edges[i])
        #     else:
        #         new_edge = embed_edges_fn_2(graph.edges[i])
            
        #     if new_edges is None:
        #         new_edges = new_edge
        #     else:
        #         new_edges = jnp.concatenate((new_edges, new_edge))
        
        # TODO: generalize to other circuits. only for LC circuit now
        new_edges = jnp.concatenate((embed_edge_fn_1(graph.edges[0]), 
                                     embed_edge_fn_2(graph.edges[1]), embed_edge_fn_1(graph.edges[2]), embed_edge_fn_2(graph.edges[3]), embed_edge_fn_1(graph.edges[4])))

        return graph._replace(nodes=embed_nodes_fn(graph.nodes),
                              edges=new_edges,
                              globals=embed_globals_fn(graph.globals))    
    return Embed