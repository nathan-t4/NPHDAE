import jax
import jraph
import jax.numpy as jnp
from utils.gnn_utils import *

def explicit_batch_graphs(graphs, Alambda, system_configs, senders, receivers):
    """ Batch graphs to get graph """
    label_encoder = lambda arr : jnp.unique(arr, return_inverse=True)[1]

    num_subsystems = len(system_configs)
    num_nodes = jnp.array([(cfg['num_nodes']) for cfg in system_configs])
    num_caps = jnp.array([cfg['num_capacitors'] for cfg in system_configs])
    num_res = jnp.array([cfg['num_resistors'] for cfg in system_configs])
    num_inds = jnp.array([cfg['num_inductors'] for cfg in system_configs])
    num_volts = jnp.array([cfg['num_volt_sources'] for cfg in system_configs])
    num_curs = jnp.array([cfg['num_cur_sources'] for cfg in system_configs])
    num_edges = jnp.array(
        [nc+nr+nl+nv+ni for (nc,nr,nl,nv,ni) in zip(num_caps, num_res, num_inds, num_volts, num_curs)]
        )

    n_node = sum(num_nodes)
    n_edge = sum(num_edges)
    qs = []
    rs = []
    phis = []
    es = []
    jvs = []

    for i, graph in enumerate(graphs):
        q = graph.edges[0 : num_caps[i]]
        r = graph.edges[num_caps[i] : num_caps[i]+num_res[i]]
        phi = graph.edges[num_caps[i]+num_res[i] : num_caps[i]+num_res[i]+num_inds[i]]
        e = graph.nodes[1:]
        jv = graph.edges[num_caps[i]+num_res[i]+num_inds[i] : num_caps[i]+num_res[i]+num_inds[i]+num_volts[i]]
        ji = graph.edges[
            num_caps[i]+num_res[i]+num_inds[i]+num_volts[i] :
            num_caps[i]+num_res[i]+num_inds[i]+num_volts[i]+num_curs[i]
        ]
        qs.append(q)
        rs.append(r)
        phis.append(phi)
        es.append(e)
        jvs.append(jv)
    
    # Need to remove redundant nodes...
    es.insert(0, jnp.array([0]))

    new_edges = jnp.concatenate((qs, rs, phis, jvs, jis))
    new_nodes = jnp.concatenate(es) # Need to remove redundant nodes

    node_fis = jnp.cumsum(num_nodes)
    node_iis = jnp.r_[jnp.array([0]), node_fis[:-1]]
    node_idx = [jnp.arange(ni, nf) for ni, nf in zip(node_iis, node_fis)]
    # Look at Alambda to decide which nodes to equate
    equivalent_nodes = [jnp.where(col != 0)[0] for col in Alambda.T]
    for node_pairs in equivalent_nodes:
        first_idx = min(node_pairs)
        second_idx = max(node_pairs)
        for i in range(num_subsystems):
            node_idx[i] = node_idx[i].at[jnp.where(node_idx[i] == first_idx)].set(second_idx)

    graph = jraph.GraphsTuple(nodes=new_nodes,
                              edges=new_edges,
                              globals=None,
                              n_node=jnp.array([n_node]),
                              n_edge=jnp.array([n_edge]),
                              senders=senders,
                              receivers=receivers)
    
    return graph


def explicit_unbatch_graph(graph, Alambda, system_configs):
    label_encoder = lambda arr : jnp.unique(arr, return_inverse=True)[1]

    num_subsystems = len(system_configs)
    num_nodes = jnp.array([(cfg['num_nodes']) for cfg in system_configs])
    num_caps = jnp.array([cfg['num_capacitors'] for cfg in system_configs])
    num_res = jnp.array([cfg['num_resistors'] for cfg in system_configs])
    num_inds = jnp.array([cfg['num_inductors'] for cfg in system_configs])
    num_volts = jnp.array([cfg['num_volt_sources'] for cfg in system_configs])
    num_curs = jnp.array([cfg['num_cur_sources'] for cfg in system_configs])

    node_fis = jnp.cumsum(num_nodes)
    node_iis = jnp.r_[jnp.array([0]), node_fis[:-1]]
    node_idx = [jnp.arange(ni, nf) for ni, nf in zip(node_iis, node_fis)]
    # Look at Alambda to decide which nodes to equate
    equivalent_nodes = [jnp.where(col != 0)[0] for col in Alambda.T]
    for node_pairs in equivalent_nodes:
        first_idx = min(node_pairs)
        second_idx = max(node_pairs)
        for i in range(num_subsystems):
            node_idx[i] = node_idx[i].at[jnp.where(node_idx[i] == first_idx)].set(second_idx)

    # Concatenate node_idx to apply label_encoder. 
    # Then, split back to get list of node_idxs for each subsystem (splitting idxs = jnp.cumsum(num_nodes))
    node_idx = jnp.split(
        label_encoder(jnp.concatenate(node_idx)), jnp.cumsum(num_nodes)[:-1]
        )
    
    
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

    graph_nodes_wo_gnd = graph.nodes[1:,0]
    nodes = [
        graph_nodes_wo_gnd[n_idx]
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
        # Append ground node
        node_i = jnp.concatenate((jnp.array([0]), nodes[i]))

        graph_i = jraph.GraphsTuple(
            nodes=node_i,
            edges=edge_i,
            globals=None, # TODO: make a globals[i]
            receivers=receiver_i,
            senders=sender_i,
            n_node=jnp.array([len(node_i)]),
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

def get_composite_system_config(system_configs, Alambda):
        ACs = [cfg['AC'] for cfg in system_configs]
        ARs = [cfg['AR'] for cfg in system_configs]
        ALs = [cfg['AL'] for cfg in system_configs]
        AVs = [cfg['AV'] for cfg in system_configs]
        AIs = [cfg['AI'] for cfg in system_configs]

        num_nodes = [cfg['num_nodes'] for cfg in system_configs]
        num_capacitors = [cfg['num_capacitors'] for cfg in system_configs]
        num_resistors = [cfg['num_resistors'] for cfg in system_configs]
        num_inductors = [cfg['num_inductors'] for cfg in system_configs]
        num_volt_sources = [cfg['num_volt_sources'] for cfg in system_configs]
        num_cur_sources = [cfg['num_cur_sources'] for cfg in system_configs]
        state_dims = [cfg['state_dim'] for cfg in system_configs]
        num_lamb = len(Alambda.T)

        ncc = sum(num_capacitors)
        nrc = sum(num_resistors)
        nlc = sum(num_inductors)
        nvc = sum(num_volt_sources)
        nic = sum(num_cur_sources)
        # TODO: avoid recounting gnd
        nec = sum(num_nodes)
        state_dim_c = ncc+nlc+nec+nvc+num_lamb

        comp_AC = jax.scipy.linalg.block_diag(*[AC for AC in ACs])
        comp_AR = jax.scipy.linalg.block_diag(*[AR for AR in ARs])
        comp_AL = jax.scipy.linalg.block_diag(*[AL for AL in ALs])
        comp_AV = jax.scipy.linalg.block_diag(*[AV for AV in AVs])
        comp_AI = jax.scipy.linalg.block_diag(*[AI for AI in AIs])

        comp_AC = nonzero_columns(comp_AC)
        comp_AR = nonzero_columns(comp_AR)
        comp_AL = nonzero_columns(comp_AL)
        comp_AV = nonzero_columns(comp_AV)
        comp_AI = nonzero_columns(comp_AI)

        # Create composite J matrix
        J =  jnp.zeros((state_dim_c, state_dim_c))
        J = J.at[0:nec, nec:nec+nlc].set(-comp_AL)
        J = J.at[0:nec, nec+nlc+ncc : nec+nlc+ncc+nvc].set(-comp_AV)
        J = J.at[0:nec, nec+nlc+ncc+nvc : nec+nlc+ncc+nvc+num_lamb].set(-Alambda)
        
        J = J.at[nec : nec+nlc, 0:nec].set(comp_AL.T)
        J = J.at[nec+nlc+ncc : nec+nlc+ncc+nvc, 0:nec].set(comp_AV.T)
        J = J.at[nec+nlc+ncc+nvc : nec+nlc+ncc+nvc+num_lamb, 0:nec].set(Alambda.T)

        # Create composite E matrix
        E = jnp.zeros((state_dim_c, state_dim_c))
        E = E.at[0:nec, 0:ncc].set(comp_AC)
        E = E.at[nec:nec+nlc, ncc:ncc+nlc].set(jnp.eye(nlc))

        # Create composite r vector 
        def get_composite_r(z):
            g = lambda e : (comp_AR.T @ e) / 1.0 
            # z = [e, jl, uc, jv, lamb]
            e = z[0 : nec]
            uc = z[nec+nlc : nec+nlc+ncc]

            curr_through_resistors = jnp.linalg.matmul(comp_AR, g(e))
            charge_constraint = jnp.matmul(comp_AC.T, e) - uc

            diss = jnp.zeros((state_dim_c,))
            diss = diss.at[0:nec].set(curr_through_resistors)
            diss = diss.at[(nec+nlc):(nec+nlc+ncc)].set(charge_constraint)

            return diss
        
        r = get_composite_r

        # Create composite B_bar
        B_bar = jnp.zeros((state_dim_c, nic+nvc))
        B_bar = B_bar.at[0:nec, 0:nic].set(-comp_AI)
        B_bar = B_bar.at[nec+nlc+ncc : nec+nlc+ncc+nvc, nic : nic+nvc].set(-jnp.eye(nvc))

        # Find the indices corresponding to the differential and algebraic variables
        diff_indices, alg_indices = get_diff_and_alg_indices(E)
        alg_eq_indices = get_alg_eq_indices(E)
        num_diff_vars = len(diff_indices)
        num_alg_vars = len(alg_indices) # avoid recounting gnd

        # LU decomposition on composite E + get inverses of decomposition matrices
        P, L, U = jax.scipy.linalg.lu(E)
        P_inv = jax.scipy.linalg.inv(P)
        L_inv = jax.scipy.linalg.inv(L)
        U_inv = jax.scipy.linalg.inv(U[diff_indices,:][:,diff_indices])

        comp_net_config = {
            'E': E,
            'J': J,
            'r': r,
            'B': B_bar,
            'diff_indices': diff_indices,
            'alg_indices': alg_indices,
            'alg_eq_indices': alg_eq_indices,
            'num_diff_vars': num_diff_vars,
            'num_alg_vars': num_alg_vars,
            'P_inv': P_inv,
            'L_inv': L_inv,
            'U_inv': U_inv,
        }

        return comp_net_config
