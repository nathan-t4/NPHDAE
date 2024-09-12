import jax
import jraph
import jax.numpy as jnp
from copy import deepcopy
from utils.gnn_utils import *

def delete_equivalent_nodes(nodes, num_nodes, Alambda):
    node_fis = jnp.cumsum(jnp.asarray(num_nodes))
    node_iis = jnp.r_[jnp.array([0]), node_fis[:-1]]
    node_idx = [jnp.arange(ni, nf) for ni, nf in zip(node_iis, node_fis)]

    # Look at Alambda to decide which nodes to equate and delete
    equivalent_nodes = [jnp.where(col != 0)[0] for col in Alambda.T]
    nodes_to_remove = jnp.sort(
        jnp.asarray([x if x in node_idx[1] else y for (x,y) in equivalent_nodes])
    )
    for i in reversed(nodes_to_remove):
        nodes = jnp.delete(nodes, i)
    
    return nodes

def delete_equivalent_nodes_exp(nodes, num_nodes, Alambda):
    nodes = nodes[jnp.array([0,1,2,4,6,7,8])]
    return nodes

def explicit_batch_graphs(graphs, Alambda, system_configs, senders, receivers):
    """ Batch graphs to get graph """
    num_subsystems = len(system_configs)
    num_nodes = jnp.array([(cfg['num_nodes']) for cfg in system_configs])
    num_caps = jnp.array([cfg['num_capacitors'] for cfg in system_configs])
    num_res = jnp.array([cfg['num_resistors'] for cfg in system_configs])
    num_inds = jnp.array([cfg['num_inductors'] for cfg in system_configs])
    num_volts = jnp.array([cfg['num_volt_sources'] for cfg in system_configs])
    num_curs = jnp.array([cfg['num_cur_sources'] for cfg in system_configs])

    qs = []
    rs = []
    phis = []
    es = []
    jvs = []
    jis = []
    globals = []

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
        es.append(e)
        qs.append(q)
        rs.append(r)
        phis.append(phi)
        jvs.append(jv)
        jis.append(ji)
        globals.append(graph.globals)

    es = jnp.concatenate(es) # without ground node
    qs = jnp.concatenate(qs)
    rs = jnp.concatenate(rs)
    phis = jnp.concatenate(phis)
    jvs = jnp.concatenate(jvs)
    jis = jnp.concatenate(jis)
    globals = jnp.concatenate(globals)

    
    # Need to remove redundant nodes...
    edges = jnp.concatenate((qs, rs, phis, jvs, jis))
    nodes = es

    # node_fis = jnp.cumsum(num_nodes)
    # node_iis = jnp.r_[jnp.array([0]), node_fis[:-1]]
    # node_idx = [jnp.arange(ni, nf) for ni, nf in zip(node_iis, node_fis)]
    # # Look at Alambda to decide which nodes to equate and delete
    # equivalent_nodes = [jnp.where(col != 0)[0] for col in Alambda.T]
    # nodes_to_remove = jnp.sort([x if x in node_idx[1] else y for (x,y) in equivalent_nodes])
    # for i in reversed(nodes_to_remove):
    #     nodes = jnp.delete(nodes, i)

    nodes = delete_equivalent_nodes_exp(nodes, num_nodes, Alambda)

    # Append ground node
    nodes = jnp.concatenate(([0],nodes)).reshape(-1,1)
    
    graph = jraph.GraphsTuple(nodes=nodes,
                              edges=edges,
                              globals=globals,
                              n_node=jnp.array([nodes.shape[0]]),
                              n_edge=jnp.array([edges.shape[0]]),
                              senders=senders,
                              receivers=receivers)
    
    return graph

def get_subsystem_node_indices(num_nodes, Alambda):
    label_encoder = lambda arr : jnp.unique(arr, return_inverse=True)[1]
    num_nodes = jnp.asarray(num_nodes)
    num_subsystems = len(num_nodes)
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

    original_node_idx_wo_overlap = jnp.unique(jnp.concatenate(node_idx))

    # Concatenate node_idx to apply label_encoder. 
    # Then, split back to get list of node_idxs for each subsystem (splitting idxs = jnp.cumsum(num_nodes))
    node_idx = jnp.split(
        label_encoder(jnp.concatenate(node_idx)), jnp.cumsum(num_nodes)[:-1]
        )
    
    return node_idx

def get_subsystem_node_indices_exp(num_nodes, Alambda):
    return jnp.array([[0,1,2], [2,3,6], [4,5,6]])


def explicit_unbatch_graph(graph, Alambda, system_configs):
    label_encoder = lambda arr : jnp.unique(arr, return_inverse=True)[1]

    num_subsystems = len(system_configs)
    num_nodes = jnp.array([(cfg['num_nodes']) for cfg in system_configs])
    num_caps = jnp.array([cfg['num_capacitors'] for cfg in system_configs])
    num_res = jnp.array([cfg['num_resistors'] for cfg in system_configs])
    num_inds = jnp.array([cfg['num_inductors'] for cfg in system_configs])
    num_volts = jnp.array([cfg['num_volt_sources'] for cfg in system_configs])
    num_curs = jnp.array([cfg['num_cur_sources'] for cfg in system_configs])

    # node_idx = get_subsystem_node_indices(num_nodes, Alambda)
    node_idx = get_subsystem_node_indices_exp(num_nodes, Alambda)
    
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
        receiver_i = label_encoder(jnp.array(receiver_i)).flatten()
        sender_i = label_encoder(jnp.array(sender_i)).flatten()
        # Append ground node
        node_i = jnp.concatenate((jnp.array([0]), nodes[i])).reshape(-1,1)

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

    num_subsystems = len(system_configs)
    num_nodes = [cfg['num_nodes'] for cfg in system_configs]
    num_capacitors = [cfg['num_capacitors'] for cfg in system_configs]
    num_resistors = [cfg['num_resistors'] for cfg in system_configs]
    num_inductors = [cfg['num_inductors'] for cfg in system_configs]
    num_volt_sources = [cfg['num_volt_sources'] for cfg in system_configs]
    num_cur_sources = [cfg['num_cur_sources'] for cfg in system_configs]
    state_dims = [cfg['state_dim'] for cfg in system_configs]
    num_lamb = len(Alambda.T)
    diff_indices = [cfg['diff_indices'] for cfg in system_configs]
    alg_indices = [cfg['alg_indices'] for cfg in system_configs]

    ncc = sum(num_capacitors)
    nrc = sum(num_resistors)
    nlc = sum(num_inductors)
    nvc = sum(num_volt_sources)
    nic = sum(num_cur_sources)
    nec = sum(num_nodes) - num_lamb
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

    # Remove equivalent nodes (as determined by Alambda) from composite incidence matrices
    def remove_equivalent_nodes_from_Alambda(A):
        equivalent_nodes = []
        row_to_remove = []
        for i in range(Alambda.shape[1]):
            equivalent_nodes.append(jnp.where(Alambda[:,i] != 0)[0])

        for node_idx in equivalent_nodes:
            # Assumes that each equivalent_node is a pair
            if (A[node_idx[0], :] == 0).all():
                row_to_remove.append(node_idx[0])
            else:
                row_to_remove.append(node_idx[1])

        row_to_keep = jnp.arange(A.shape[0])
        row_to_keep = jnp.delete(row_to_keep, jnp.array(row_to_remove))

        return A[row_to_keep]
        

    # TODO: look at equivalent nodes (e1, e2), check if which row (e1 or e2) is all zeros and remove that row.
    comp_AC = remove_equivalent_nodes_from_Alambda(comp_AC)
    comp_AR = remove_equivalent_nodes_from_Alambda(comp_AR)
    comp_AL = remove_equivalent_nodes_from_Alambda(comp_AL)
    comp_AV = remove_equivalent_nodes_from_Alambda(comp_AV)
    comp_AI = remove_equivalent_nodes_from_Alambda(comp_AI)

    # Create composite J matrix
    J =  jnp.zeros((state_dim_c, state_dim_c))
    J = J.at[0:nec, nec:nec+nlc].set(-comp_AL)
    J = J.at[0:nec, nec+nlc+ncc : nec+nlc+ncc+nvc].set(-comp_AV)
    J = J.at[0:nec+num_lamb, nec+nlc+ncc+nvc : nec+nlc+ncc+nvc+num_lamb].set(-Alambda)
    
    J = J.at[nec : nec+nlc, 0:nec].set(comp_AL.T)
    J = J.at[nec+nlc+ncc : nec+nlc+ncc+nvc, 0:nec].set(comp_AV.T)
    J = J.at[nec+nlc+ncc+nvc : nec+nlc+ncc+nvc+num_lamb, 0:nec+num_lamb].set(Alambda.T)

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
    # diff_indices, alg_indices = get_diff_and_alg_indices(E)
    diff_indices = jnp.arange(ncc+nlc)
    alg_indices = jnp.arange(ncc+nlc, state_dim_c)
    alg_eq_indices = get_alg_eq_indices(E)
    num_diff_vars = len(diff_indices)
    num_alg_vars = len(alg_indices)

    # LU decomposition on composite E + get inverses of decomposition matrices
    P, L, U = jax.scipy.linalg.lu(E)
    P_inv = jax.scipy.linalg.inv(P)
    L_inv = jax.scipy.linalg.inv(L)
    U_inv = jax.scipy.linalg.inv(U[diff_indices,:][:,diff_indices])

    # Get Alambdas
    Alambdas = [
        Alambda[sum(num_nodes[0:i]) : sum(num_nodes[0:i+1])]
        for i in range(num_subsystems)
        ]
    
    # Decompose composite system state to subsystem states

    def subsystem_to_composite_state(states, Alambda):
        qs = []; phis = []; es = []; jvs = []
        for state, nc, ni, ne, nv in zip(
            states, num_capacitors, num_inductors, num_nodes, num_volt_sources):
            qs.append(state[0 : nc])
            phis.append(state[nc : nc+ni])
            es.append(state[nc+ni : nc+ni+ne])
            jvs.append(state[nc+ni+ne : nc+ni+ne+nv])
        
        q = jnp.concatenate(qs)
        phi = jnp.concatenate(phis)
        e = jnp.concatenate(es) 
        jv = jnp.concatenate(jvs)
        # Delete equivalent nodes based on Alambda
        e = delete_equivalent_nodes_exp(e, num_nodes, Alambda)

        state = jnp.concatenate((q, phi, e, jv))

        return state, (qs, phis, es, jvs)
    
    def composite_to_subsystem_states(state):
        nc = sum(num_capacitors)
        nl = sum(num_inductors)
        ne = sum(num_nodes) - num_lamb

        # node_idx = get_subsystem_node_indices(num_nodes, Alambda)
        node_idx = get_subsystem_node_indices_exp(num_nodes, Alambda)

        states = []
        qs = []; phis = []; es = []; jvs = []
        for i in range(num_subsystems):
            q_i = state[sum(num_capacitors[:i]) : sum(num_capacitors[:i+1])]
            phi_i = state[
                nc+sum(num_inductors[:i]) : 
                nc+sum(num_inductors[:i+1])
            ]
            e_i = state[nc+nl+node_idx[i]]
            jv_i = state[
                nc+nl+ne+sum(num_volt_sources[:i]) : 
                nc+nl+ne+sum(num_volt_sources[:i+1])
            ]
            qs.append(q_i)
            phis.append(phi_i)
            es.append(e_i)
            jvs.append(jv_i)

            state_i = jnp.concatenate((q_i, phi_i, e_i, jv_i))
            states.append(state_i)

        return states, (qs, phis, es, jvs)            
    
    def get_coupling_input(lamb, es, system_k_idx):
        # Jacobian-type approach
        u_hats = []
        for i in range(num_subsystems):
            if i == system_k_idx:
                Alambdas_without_k = deepcopy(Alambdas)
                es_without_k = deepcopy(es)
                Alambdas_without_k.pop(system_k_idx)
                es_without_k.pop(system_k_idx)

                coupling_constraint = jnp.sum(
                    jnp.array([jnp.matmul(Al_i.T, e_i) for Al_i, e_i in zip(Alambdas_without_k, es_without_k)]),
                    axis=1,
                )
                
                u_hats.append(coupling_constraint)
            else:
                u_hats.append(-lamb)

        return u_hats

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
        'Alambdas': Alambdas,
        'subsystem_to_composite_state': subsystem_to_composite_state,
        'composite_to_subsystem_states': composite_to_subsystem_states,
        'get_coupling_input': get_coupling_input,
    }

    return comp_net_config
