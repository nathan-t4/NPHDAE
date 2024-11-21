import jax

import haiku as hk

import sys
sys.path.append('../')
from models.ph_dae import PHDAE
import sacred

from models.common import get_params_struct, choose_nonlinearity
import jax.numpy as jnp
from helpers.model_factory import get_model_factory
from jax.experimental.ode import odeint


class DGU_PHNDAE():

    def __init__(
            self,
            rng_key : jax.random.PRNGKey,
            model_setup : dict,
        ):
        """
        Constructor for the DGU PHNDAE.

        Parameters
        ----------
        rng_key : 
            A key for random initialization of the parameters of the 
            neural networks.
        model_setup : 
            Dictionary containing the setup details for the model.
        """

        self.dt = model_setup['dt']
        self.rng_key = rng_key
        self.init_rng_key = rng_key
        self.input_dim = model_setup['input_dim']
        self.output_dim = model_setup['output_dim']

        self.AC = jnp.array(model_setup['AC'])
        self.AR = jnp.array(model_setup['AR'])
        self.AL = jnp.array(model_setup['AL'])
        self.AV = jnp.array(model_setup['AV'])
        self.AI = jnp.array(model_setup['AI'])

        self.R = jnp.array(model_setup['R'])
        self.L = jnp.array(model_setup['L'])
        self.C = jnp.array(model_setup['C'])
        
        # Testing parameters
        # self.scalings = jnp.array(model_setup['scalings']) if 'scalings' in model_setup.keys() else jnp.ones(self.output_dim)
        if 'stds' in model_setup.keys():
            self.stds = jnp.array(model_setup['stds'])
        else:
            self.stds = jnp.ones(self.output_dim)

        if 'means' in model_setup.keys():
            self.means = jnp.array(model_setup['means'])
        else:
            self.means = jnp.zeros(self.output_dim)

        self.one_timestep_solver = model_setup['one_timestep_solver'] if 'one_timestep_solver' in model_setup.keys() else 'rk4'
        
        self.model_setup = model_setup.copy()

        self._get_num_vars()
        self._get_scaling()
        self._build_ph_ndae()
        self.params_shapes, self.count, self.params_tree_struct = \
            get_params_struct(self.init_params)

    def _get_num_vars(self):
        """
        Get the number of variables of each type in the system.
        """
        assert self.AC.shape[0] == self.AR.shape[0] == self.AL.shape[0] == self.AV.shape[0] == self.AI.shape[0], "All matrices must have the same number of rows."

        self.num_nodes = self.AC.shape[0] # number of non-ground nodes
        self.num_capacitors = self.AC.shape[1]
        self.num_inductors = self.AL.shape[1]
        self.num_voltage_sources = self.AV.shape[1]
        self.num_current_sources = self.AI.shape[1]
        self.num_resistors = self.AR.shape[1]

        if (self.AC == 0.0).all():
            self.num_capacitors = 0
        if (self.AR == 0.0).all():
            self.num_resistors = 0
        if (self.AL == 0.0).all():
            self.num_inductors = 0
        if (self.AV == 0.0).all():
            self.num_voltage_sources = 0
        if (self.AI == 0.0).all():
            self.num_current_sources = 0

        # There is a differential variable for each capacitor and inductor in the circuit
        self.num_differential_vars = self.num_capacitors + self.num_inductors

        # There is an algebraic variable for the voltage at each node, and for the current through each voltage source
        self.num_algebraic_vars = self.num_nodes + self.num_voltage_sources 

    def _get_scaling(self):
        q_mean = self.means[0 : self.num_capacitors]
        q_std = self.stds[0 : self.num_capacitors]

        q_mean = jnp.mean(q_mean)
        q_std = jnp.mean(q_std)

        phi_mean = self.means[self.num_capacitors : self.num_capacitors+self.num_inductors]
        phi_std = self.stds[self.num_capacitors : self.num_capacitors+self.num_inductors]

        phi_mean = jnp.mean(phi_mean)
        phi_std = jnp.mean(phi_std)

        e_mean = self.means[self.num_capacitors + self.num_inductors : self.num_capacitors + self.num_inductors + self.num_nodes]
        e_std = self.stds[self.num_capacitors + self.num_inductors : self.num_capacitors + self.num_inductors + self.num_nodes]

        e_mean = jnp.mean(e_mean)
        e_std = jnp.mean(e_std)
        
        # self.grad_H_func_scale = (phi_mean, phi_std)
        self.grad_H_func_scale = (0.0, 1.0)
        self.r_func_scale = (e_mean, e_std, 0.0, 1.0)
        self.q_func_scale = (e_mean, e_std, 0.0, 1.0)

    def _build_ph_ndae(self):

        init_params = {}

        # Define the H function for the inductors
        self.rng_key, subkey = jax.random.split(self.rng_key)
        H_net = get_model_factory(self.model_setup['H_net_setup']).create_model(subkey)
        init_params['grad_H_func'] = H_net.init_params

        num_inductors = self.num_inductors
        def grad_H_func(phi, params=None, scales=self.grad_H_func_scale):
            mean, std = scales
            def H_forward(x):
                H = H_net.forward(params=params, x=x)
                return H.squeeze()
                # return 0.5 * (x.squeeze()**2) / self.L
            gradH = jax.grad(H_forward)((phi - mean) / std) * std + mean
            # gradH = H_forward(phi / scale) * scale
            gradH = gradH.reshape((num_inductors,))
            return gradH

        # self.H_net = H_net
        self.grad_H_func = jax.jit(grad_H_func)

        # Define the R function for the resistors
        self.rng_key, subkey = jax.random.split(self.rng_key)
        r_net = get_model_factory(self.model_setup['r_net_setup']).create_model(subkey)
        init_params['r_func'] = r_net.init_params

        num_resistors = self.num_resistors
        def r_func(delta_V, params=None, scales=self.r_func_scale):
            mean_in, std_in, mean_out, std_out = scales
            def R_forward(x):
                R = r_net.forward(params=params, x=x)
                return R.squeeze()
                # return x.squeeze() / self.R
            jR = R_forward((delta_V - mean_in) / std_in) * std_out + mean_out
            jR = jR.reshape((num_resistors,))
            return jR
        self.r_func = jax.jit(r_func)
    
        # Define the Q function for the capacitors
        self.rng_key, subkey = jax.random.split(self.rng_key)
        q_net = get_model_factory(self.model_setup['q_net_setup']).create_model(subkey)
        init_params['q_func'] = q_net.init_params

        num_capacitors = self.num_capacitors
        def q_func(delta_V, params=None, scales=self.q_func_scale):  
            """ Return the charge q_C across the capacitor """ 
            mean_in, std_in, mean_out, std_out = scales
            def Q_forward(x):
                Q = q_net.forward(params=params, x=x)
                return Q.squeeze()
                # return self.C * x.squeeze()
            q_C = Q_forward((delta_V - mean_in) / std_in) * std_out + mean_out
            q_C = q_C.reshape((num_capacitors,))
            return q_C
        self.q_func = jax.jit(q_func)

        def u_func(t, params):
            return jnp.array(params)

        self.u_func = jax.jit(u_func)
        init_params['u_func_params'] = None # Don't make frequency a parameter here, otherwise training will try and optimize it.

        self.dae = PHDAE(
            self.AC, 
            self.AR, 
            self.AL, 
            self.AV, 
            self.AI, 
            self.grad_H_func,
            self.q_func, 
            self.r_func, 
            self.u_func,
            self.one_timestep_solver,
            # 'implicit_trapezoid'
        )

        def forward(params, z, u):
            t = z[-1]
            z = z[:-1]
            params['u_func'] = u
            return self.dae.solver.one_timestep_solver(z, t, self.dt, params)
        
        self.forward = jax.jit(forward)
        self.forward = jax.vmap(forward, in_axes=(None, 0, 0))

        def forward_g(params, z, u):
            t = z[-1]
            z = z[:-1]

            x = z[0:self.num_differential_vars]
            y = z[self.num_differential_vars::]
            g = self.dae.g

            params['u_func'] = u

            return g(x, y, t, params)
        
        self.forward_g = jax.jit(forward_g)
        self.forward_g = jax.vmap(forward_g, in_axes=(None, 0, 0))
        self.init_params = init_params