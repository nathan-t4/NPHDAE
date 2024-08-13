from scripts.graph_builders.gb_base import *
from utils.gnn_utils import graph_from_incidence_matrices

@register_pytree_node_class
class TestGraphBuilder(GraphBuilder):
    def __init__(self, path, AC, AR, AL, AV, AI):
        super().__init__(path, AC, AR, AL, AV, AI, add_undirected_edges=False, add_self_loops=False)

    def _load_data(self, path):
        data = np.load(path, allow_pickle=True)
        config = data['config']
        state = data['state_trajectories']
        u = data['control_inputs']
        self._num_trajectories = state.shape[0]
        self._num_timesteps = state.shape[1]
        self._dt = config['dt']
        self.R = 1.0
        self.L = 1.0
        self.C = 1.0
        self.system_params = {
            'R': self.R,
            'L': self.L,
            'C': self.C,
        }
        self._control = jnp.array(u)
        self._Qs = jnp.array(state[:,:,0:self.num_capacitors])
        self._Phis = jnp.array(
            state[:,:,self.num_capacitors:self.num_capacitors+self.num_inductors]
        )
        ground_node = jnp.zeros((self._num_trajectories, self._num_timesteps, 1))
        other_nodes = jnp.array(
            state[:,:, 
                  self.num_capacitors+self.num_inductors : 
                  self.num_capacitors+self.num_inductors+(self.num_nodes-1)]
        )
        # Append ground node
        self._Vs = jnp.concatenate((ground_node, other_nodes), axis=-1) 
        self._jv = jnp.array(state[:,:,self.num_capacitors+self.num_inductors+(self.num_nodes-1) : self.num_capacitors+self.num_inductors+(self.num_nodes-1)+self.num_volt_sources])
        self._is = self._control[:,:,0:self.num_cur_sources]

        self._H = 0.5 * (self._Qs**2 / self.C + self._Phis**2 / self.L)
        self._residuals = jnp.zeros((self._num_trajectories, self._num_timesteps, 1))
        self.edge_idxs = np.arange(
            self.num_capacitors+self.num_inductors+self.num_resistors+self.num_volt_sources+self.num_cur_sources
        ) # needs to be np instead of jnp
        # self.include_idxs = jnp.array([0, 0, 1, 1])
        self.include_idxs = None
        self.node_idxs = None
        self.differential_vars = jnp.arange(self.num_capacitors+self.num_inductors)
        self.algebraic_vars = jnp.arange(self.num_capacitors+self.num_inductors, self._num_states)

        self.senders, self.receivers = graph_from_incidence_matrices(
            (self.AC, self.AR, self.AL, self.AV, self.AI)
            )

    def get_control(self, trajs, ts):
        return self._control[trajs, ts]
    
    def get_pred_data(self, graph):
        pred_Qs = graph.edges[0:self.num_capacitors,0]
        pred_Phis = graph.edges[
            self.num_capacitors+self.num_resistors : 
            self.num_capacitors+self.num_resistors+self.num_inductors, 0]
        pred_Vs = graph.nodes[:,0]
        pred_jv = graph.edges[
            self.num_capacitors+self.num_resistors+self.num_inductors : 
            self.num_capacitors+self.num_resistors+self.num_inductors+self.num_volt_sources, 0]
        pred_is = graph.edges[
            self.num_capacitors+self.num_resistors+self.num_inductors+self.num_volt_sources : 
            self.num_capacitors+self.num_resistors+self.num_inductors+self.num_volt_sources+self.num_cur_sources, 0]
        pred_diff_states = jnp.concatenate((pred_Qs, pred_Phis), axis=-1).squeeze()
        pred_alg_states = jnp.concatenate((pred_Vs, pred_jv), axis=-1).squeeze()
        pred_H = jnp.array([graph.globals[0]])
        residual = jnp.array([jnp.sum((graph.globals[1:]))])
        return (pred_diff_states, pred_alg_states, pred_H, residual) 
    
    def get_batch_pred_data(self, graphs) -> Sequence[jraph.GraphsTuple]:
        def f(carry, graph):
            return carry, self.get_pred_data(graph)
        
        _, batch_data = jax.lax.scan(f, None, graphs)
        
        return batch_data

    def get_exp_data(self, trajs, ts) -> Tuple:
        exp_diff_states = jnp.concatenate((self._Qs[trajs, ts], self._Phis[trajs, ts]), axis=-1).squeeze()
        exp_alg_states = jnp.concatenate((self._Vs[trajs, ts], self._jv[trajs, ts]), axis=-1).squeeze()
        exp_H = self._H[trajs, ts]
        exp_residuals = self._residuals[trajs, ts]
        return (exp_diff_states, exp_alg_states, exp_H, exp_residuals)
    
    def _get_norm_stats(self):
        # TODO
        norm_stats = ml_collections.ConfigDict()
        self._norm_stats = norm_stats
    
    def _setup_graph_params(self):
        pass

    def get_graph_from_state(self, state, control=None, system_params=None, set_nodes=False, set_ground_and_control=False, nodes=None, globals=None) -> jraph.GraphsTuple:
        Q = state[0 : self.num_capacitors]
        Phi = state[self.num_capacitors : self.num_capacitors+self.num_inductors]
        e = state[
            self.num_capacitors+self.num_inductors : 
            self.num_capacitors+self.num_inductors+self.num_nodes
        ]
        jv = state[
            self.num_capacitors+self.num_inductors+self.num_nodes :
            self.num_capacitors+self.num_inductors+self.num_nodes+self.num_volt_sources
        ]
        I = control[0 : self.num_cur_sources]
        V = control[self.num_cur_sources : self.num_cur_sources+self.num_volt_sources]

        nodes = e.reshape(-1,1)
        resistor_current = jnp.matmul(self.AR.T, e) # This is g
        # resistor_current = jv # TODO: trial 
        edges = []
        if self.num_capacitors > 0:
            edges.append(Q)
        if self.num_resistors > 0:
            edges.append(resistor_current)
        if self.num_inductors > 0:
            edges.append(Phi)
        if self.num_volt_sources > 0:
            edges.append(jv)
        if self.num_cur_sources > 0:
            edges.append(I)
        edges = jnp.stack(edges)
      # edges = jnp.stack((Q, resistor_current, Phi, jv, I))

        if set_nodes:
            raise NotImplementedError()
        if set_ground_and_control:
            # TODO: automate this
            # nodes = nodes.at[0].set(jnp.array([0]))
            # nodes = nodes.at[]

            # Set voltage source
            nodes = jnp.concatenate((jnp.array([[0]]), jnp.array([V]), nodes[2:]), axis=0)
            # Set current sources
            if self.num_cur_sources > 0:
                edges = edges.at[-self.num_cur_sources:, 0].set(I)

        edges = jnp.concatenate((edges, self.edge_idxs.reshape(-1,1)), axis=1)

        n_node = self.n_node
        n_edge = self.n_edge
        senders = self.senders
        receivers = self.receivers
        global_context = globals

        graph = jraph.GraphsTuple(
            nodes=nodes,
            edges=edges,
            globals=global_context,
            n_node=n_node,
            n_edge=n_edge,
            senders=senders,
            receivers=receivers,
        )

        return graph
    
    def get_state_from_graph(self, graph):
        q = graph.edges[0 : self.num_capacitors, 0]
        phi = graph.edges[
            self.num_capacitors+self.num_resistors : 
            self.num_capacitors+self.num_resistors+self.num_inductors, 0]
        e = graph.nodes.squeeze()
        jv = graph.edges[
            self.num_capacitors+self.num_resistors+self.num_inductors :
            self.num_capacitors+self.num_resistors+self.num_inductors+self.num_volt_sources, 0]
        state = jnp.r_[q, phi, e, jv]
        return state
    
    def get_alg_vars_from_graph(self, graph):
        e = graph.nodes.squeeze()
        # If the system has voltage sources, add 'jv' as algebraic variables
        if self.num_volt_sources > 0:
            jv = graph.edges[
                self.num_capacitors+self.num_resistors+self.num_inductors :
                self.num_capacitors+self.num_resistors+self.num_inductors+self.num_volt_sources
            , 0]
            y = jnp.r_[e, jv]
        else:
            y = e
        return y
    
    def get_graph(self, traj_idx, t) -> jraph.GraphsTuple:
        nodes = jnp.array(self._Vs[traj_idx, t]).reshape(-1,1)
        edges = []
        if self.num_capacitors > 0:
            edges.append(self._Qs[traj_idx, t].squeeze())
        if self.num_resistors > 0:
            resistor_current = jnp.matmul(self.AR.T, self._Vs[traj_idx, t]) # this is g
            # resistor_current = self._jv[traj_idx, t]
            edges.append(resistor_current.squeeze())
        if self.num_inductors > 0:
            edges.append(self._Phis[traj_idx, t].squeeze())
        if self.num_volt_sources > 0:
            edges.append(self._jv[traj_idx, t].squeeze())
        if self.num_cur_sources > 0:
            edges.append(self._is[traj_idx, t].squeeze())
        
        edges = jnp.array(edges).reshape(-1,1)
        edges = jnp.concatenate((edges, self.edge_idxs.reshape(-1,1)), axis=1)
        global_context = None

        graph =  jraph.GraphsTuple(
                    nodes=nodes,
                    edges=edges,
                    senders=self.senders,
                    receivers=self.receivers,
                    n_node=self.n_node,
                    n_edge=self.n_edge,
                    globals=global_context)
        
        return graph

    def get_graph_batch(self, traj_idxs, t0s) -> Sequence[jraph.GraphsTuple]:
        def f(carry, idxs):
            return carry, self.get_graph(*idxs)
        
        _, graphs = jax.lax.scan(f, None, (traj_idxs, t0s))
        
        return graphs
    
    def get_state(self, traj_idx, t) -> jnp.ndarray:
        return jnp.array([self._Qs[traj_idx, t],
                          self._Phis[traj_idx, t],
                          self._Vs[traj_idx, t],
                          self._jv[traj_idx, t],])
    
    def get_state_batch(self, traj_idxs, t0s) -> jnp.ndarray:
        def f(carry, idxs):
            return carry, self.get_state(*idxs)
        
        _, states = jax.lax.scan(f, None, (traj_idxs, t0s))
        return states
    
    def plot(self, pred, exp, prefix=None, plot_dir='.'):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        import os

        cmap = cm.tab10

        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)

        fig = plt.figure(layout="constrained", figsize=(20,10))
        fig.suptitle(f'{prefix}')


        values = []

        values[
            0 : 
            self.num_capacitors] = [f'Q{i}' for i in range(self.num_capacitors)]    
        values[
            self.num_capacitors : 
            self.num_capacitors+self.num_inductors
            ] = [f'Phi{i}' for i in range(self.num_inductors)]
        values[
            self.num_capacitors+self.num_inductors : 
            self.num_capacitors+self.num_inductors+self.num_nodes
            ] = [f'V{i}' for i in range(self.num_nodes)]
        values[
            self.num_capacitors+self.num_inductors+self.num_nodes : 
            self.num_capacitors+self.num_inductors+self.num_nodes+self.num_volt_sources
            ] = [f'jv{i}' for i in range(self.num_volt_sources)]
        values.append('H')
        values.append('residual')

        layout = [values, [f'{v}e' for v in values]]
        ax = fig.subplot_mosaic(layout)

        ts = np.arange(self._num_timesteps) * self._dt
        for i,v in enumerate(values):
            ax[v].set_title(f'${v}$')
            ax[v].plot(ts, pred[:,i], color=cmap(0), ls='-', label=f'pred ${v}$')
            ax[v].plot(ts, exp[:,i], color=cmap(0), ls='--', label=f'exp ${v}$')
            ax[v].set_xlabel('Time [$s$]')
            ax[v].set_ylabel(f'${v}$')
            ax[v].legend()

            ax[f'{v}e'].set_title(f'${v}$ Error')
            ax[f'{v}e'].plot(ts, exp[:,i] - pred[:,i], color=cmap(0), ls='-')
            ax[f'{v}e'].set_xlabel('Time [$s$]')
            ax[f'{v}e'].set_ylabel(f'${v}$')

        plt.savefig(os.path.join(plot_dir, f'{prefix}.png'))
        plt.close()
    
    def tree_flatten(self):
        children = ()
        aux_data = (self._dt, self.R, self.L, self.C, self.system_params, self._Qs, self._Phis, self._Vs, self._jv, self._is, self._H, self._control, self._residuals, self._num_trajectories, self._num_timesteps, self._num_states, self.differential_vars, self.algebraic_vars, self.senders, self.receivers)
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        obj = object.__new__(TestGraphBuilder)
        obj._dt                   = aux_data[0]
        obj.R                     = aux_data[1]
        obj.L                     = aux_data[2]
        obj.C                     = aux_data[3]
        obj.system_params         = aux_data[4]
        obj._Qs                   = aux_data[5]
        obj._Phis                 = aux_data[6]
        obj._Vs                   = aux_data[7]
        obj._jv                   = aux_data[8]
        obj._is                   = aux_data[9]
        obj._H                    = aux_data[10]
        obj._control              = aux_data[11]
        obj._residuals            = aux_data[12]
        obj._num_trajectories     = aux_data[13]
        obj._num_timesteps        = aux_data[14]
        obj._num_states           = aux_data[15]
        obj.differential_vars     = aux_data[16]
        obj.algebraic_vars        = aux_data[17]
        obj.senders               = aux_data[18]
        obj.receivers             = aux_data[19]

        obj._setup_graph_params()
        return obj