import jraph
import jax.numpy as jnp

def explicit_unbatch_graph(graph, Alambda, system_configs):
    label_encoder = lambda arr : jnp.unique(arr, return_inverse=True)[1]

    num_subsystems = len(system_configs)
    num_nodes = jnp.array([(cfg['num_nodes']-1) for cfg in system_configs])

    num_caps = jnp.array([cfg['num_capacitors'] for cfg in system_configs])
    num_res = jnp.array([cfg['num_resistors'] for cfg in system_configs])
    num_inds = jnp.array([cfg['num_inductors'] for cfg in system_configs])
    num_volts = jnp.array([cfg['num_volt_sources'] for cfg in system_configs])
    num_curs = jnp.array([cfg['num_cur_sources'] for cfg in system_configs])

    # Look at Alambda to decide which nodes to equate
    node_fis = jnp.cumsum(num_nodes)
    node_iis = jnp.r_[jnp.array([0]), node_fis[:-1]]
    node_idx = [jnp.arange(ni, nf) for ni, nf in zip(node_iis, node_fis)]
    equivalent_nodes = [jnp.where(col != 0)[0] for col in Alambda.T]
    for node_pairs in equivalent_nodes:
        first_idx = node_pairs[0]
        second_idx = node_pairs[1]
        for i in range(len(node_idx)):
            node_idx[i] = node_idx[i].at[jnp.where(node_idx[i] == second_idx)].set(first_idx)

    idx = jnp.cumsum(jnp.array([len(arr) for arr in node_idx]))
    node_idx = jnp.split(label_encoder(jnp.concatenate(node_idx)), idx[:-1])
    
    n_cap = sum(num_caps)
    n_res = sum(num_res)
    n_ind = sum(num_inds)
    n_volt = sum(num_volts)

    cap_fis = jnp.cumsum(num_caps)
    res_fis = jnp.cumsum(num_res) + n_cap
    ind_fis = jnp.cumsum(num_inds) + n_res + n_cap
    jv_fis = jnp.cumsum(num_volts) + n_ind + n_res + n_cap
    cur_fis = jnp.cumsum(num_curs) + n_volt + n_ind + n_res + n_cap

    cap_iis = jnp.r_[jnp.array([0]), cap_fis[:-1]]
    res_iis = jnp.r_[jnp.array([n_cap]), res_fis[:-1]]
    ind_iis = jnp.r_[jnp.array([n_cap + n_res]), ind_fis[:-1]]
    jv_iis = jnp.r_[
        jnp.array([n_cap + n_res + n_ind]), jv_fis[:-1]
        ]
    cur_iis = jnp.r_[
        jnp.array([n_cap + n_res + n_ind + n_volt]), 
        cur_fis[:-1]
        ]

    graph_nodes_wo_gnd = graph.nodes[1:]

    nodes = [
        jnp.concatenate((graph.nodes[jnp.array([0])], graph_nodes_wo_gnd[n_idx]))
        for n_idx in node_idx
        ]

    # extract subsystem edges from composite system graph
    graphs = []
    for i in range(num_subsystems):
        edge_i = []
        receiver_i = []
        sender_i = []
        if num_caps[i] > 0:
            edge_i.append(graph.edges[cap_iis[i] : cap_fis[i]])
            receiver_i.append(graph.receivers[cap_iis[i] : cap_fis[i]])
            sender_i.append(graph.senders[cap_iis[i] : cap_fis[i]])
        if num_res[i] > 0:
            edge_i.append(graph.edges[res_iis[i] : res_fis[i]])
            receiver_i.append(graph.receivers[res_iis[i] : res_fis[i]])
            sender_i.append(graph.senders[res_iis[i] : res_fis[i]])
        if num_inds[i] > 0:
            edge_i.append(graph.edges[ind_iis[i] : ind_fis[i]])
            receiver_i.append(graph.receivers[ind_iis[i] : ind_fis[i]])
            sender_i.append(graph.senders[ind_iis[i] : ind_fis[i]])
        if num_volts[i] > 0:
            edge_i.append(graph.edges[jv_iis[i] : jv_fis[i]])
            receiver_i.append(graph.receivers[jv_iis[i] : jv_fis[i]])
            sender_i.append(graph.senders[jv_iis[i] : jv_fis[i]])
        if num_curs[i] > 0:
            edge_i.append(graph.edges[cur_iis[i] : cur_fis[i]])
            receiver_i.append(graph.receivers[cur_iis[i] : cur_fis[i]])
            sender_i.append(graph.senders[cur_iis[i] : cur_fis[i]])

        edge_i = jnp.concatenate(edge_i)
        receiver_i = label_encoder(jnp.array(receiver_i))
        sender_i = label_encoder(jnp.array(sender_i))

        graph_i = jraph.GraphsTuple(
            nodes=nodes[i],
            edges=edge_i,
            globals=None, # TODO: make a globals[i]
            receivers=receiver_i,
            senders=sender_i,
            n_node=jnp.array([len(nodes[i])]),
            n_edge=jnp.array([len(edge_i)]),
        )
        graphs.append(graph_i)

    return graphs

def explicit_unbatch_control(control, system_configs):
    num_subsystems = len(system_configs)
    num_cur_sources = [cfg['num_cur_sources'] for cfg in system_configs]
    num_volt_sources = [cfg['num_volt_sources'] for cfg in system_configs]
    ext_curs = [control[sum(num_cur_sources[:i]) : sum(num_cur_sources[:i+1])] for i in range(num_subsystems)]
    ext_volts = [
        control[
            sum(num_cur_sources)+sum(num_volt_sources[:i]) : 
            sum(num_cur_sources)+sum(num_volt_sources[:i+1])
            ]    
        for i in range(num_subsystems)
        ]
    
    controls = [jnp.concatenate((i, v)) for i, v in zip(ext_curs, ext_volts)]

    return controls