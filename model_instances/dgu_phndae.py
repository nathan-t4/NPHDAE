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
        self.regularization_method = model_setup['regularization_method'] if 'regularization_method' in model_setup.keys() else None
        self.reg_param = model_setup['reg_param'] if 'reg_param' in model_setup.keys() else 0.0
        self.scalings = jnp.array(model_setup['scalings']) if 'scalings' in model_setup.keys() else jnp.ones(self.output_dim)
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
        self.scaling = [self.scalings[0]] * self.num_capacitors + [self.scalings[1]] * self.num_inductors + [self.scalings[2]] * self.num_nodes + [1.0] * self.num_voltage_sources
        self.scaling = jnp.array(self.scaling)
        self.scaling = 1 / self.scaling

        # self.scalings = mean of x
        phi_scaling = self.scalings[self.num_capacitors : self.num_capacitors + self.num_inductors]
        e_scaling = self.scalings[self.num_capacitors + self.num_inductors : self.num_capacitors + self.num_inductors + self.num_nodes]
        
        self.grad_H_func_scale = jnp.mean(phi_scaling)
        self.r_func_scale = jnp.mean(e_scaling)
        self.q_func_scale = jnp.mean(e_scaling)

        # print('grad H scale', self.grad_H_func_scale)
        # print('r func scale', self.r_func_scale)
        # print('q func scale', self.q_func_scale)

    def _build_ph_ndae(self):

        init_params = {}
        init_state = {}

        # Define the H function for the inductors
        self.rng_key, subkey = jax.random.split(self.rng_key)
        H_net = get_model_factory(self.model_setup['H_net_setup']).create_model(subkey)
        init_params['grad_H_func'] = H_net.init_params
        init_state['grad_H_func'] = H_net.init_state

        num_inductors = self.num_inductors
        def grad_H_func(phi, params=None, scale=self.grad_H_func_scale):
            def H_forward(x):
                H = H_net.forward(params=params, x=x)
                # return jnp.sum(H)
                return H.squeeze()
            phi = jnp.reshape(phi, (num_inductors, 1)) / scale
            # flux = jax.vmap(jax.grad(H_forward), 0)(phi)
            flux = jax.grad(H_forward)(phi)
            flux = flux.reshape((num_inductors,)) * scale
            return flux

        self.H_net = H_net
        self.grad_H_func = jax.jit(grad_H_func)

        # Define the R function for the resistors
        self.rng_key, subkey = jax.random.split(self.rng_key)
        r_net = get_model_factory(self.model_setup['r_net_setup']).create_model(subkey)
        init_params['r_func'] = r_net.init_params
        init_state['r_func'] = r_net.init_state

        num_resistors = self.num_resistors
        # TODO: let all funcs take state as input...
        # or just change output size of this func
        def r_func(delta_V, params=None, scale=self.r_func_scale):
            def R_forward(x):
                R = r_net.forward(params=params, x=x)
                # return jnp.sum(R)
                return R.squeeze()

            # R = lambda x : x / self.R
            delta_V = jnp.reshape(delta_V, (num_resistors, 1)) / scale
            # res_cur = jax.vmap(R_forward, 0)(delta_V) # using same function on all resistors
            res_cur = R_forward(delta_V)
            res_cur = res_cur.reshape((num_resistors,)) * scale
            return res_cur
        self.r_func = jax.jit(r_func)
    
        # Define the Q function for the capacitors
        self.rng_key, subkey = jax.random.split(self.rng_key)
        q_net = get_model_factory(self.model_setup['q_net_setup']).create_model(subkey)
        init_params['q_func'] = q_net.init_params
        init_state['q_func'] = q_net.init_state

        num_capacitors = self.num_capacitors
        def q_func(delta_V, params=None, scale=self.q_func_scale):   
            def Q_forward(x):
                Q = q_net.forward(params=params, x=x)
                # return jnp.sum(Q)
                return Q.squeeze()
            # Q = lambda x : self.C * x
            delta_V = jnp.reshape(delta_V, (num_capacitors, 1)) / scale
            # q_C = jax.vmap(Q_forward, 0)(delta_V)
            q_C = Q_forward(delta_V)
            q_C = q_C.reshape((num_capacitors,)) * scale
            return q_C
        self.q_func = jax.jit(q_func)

        # voltage_source_freq = self.model_setup['u_func_freq']
        current_source_magnitude = self.model_setup['u_func_current_source_magnitude']
        voltage_source_magnitude = self.model_setup['u_func_voltage_source_magnitude']

        if current_source_magnitude is None and voltage_source_magnitude is None:
            u = jnp.array([])
        else:
            u = jnp.array([current_source_magnitude, voltage_source_magnitude])

        def u_func(t, params):
            # if params is None:
            #     return jnp.array([current_source_magnitude, voltage_source_magnitude])
            # else:
                # return jnp.array(params)
            return u

        self.u_func = jax.jit(u_func)
        # self.u_func = u_func
        init_params['u_func_params'] = None # Don't make frequency a parameter here, otherwise training will try and optimize it.
        init_state['u_func'] = None

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
            self.regularization_method,
            self.reg_param,
            self.one_timestep_solver,
            # 'implicit_trapezoid'
        )

        def forward(params, z, u):
            t = z[-1]
            z = z[:-1]
            # params['u_func'] = u
            return self.dae.solver.one_timestep_solver(z, t, self.dt, params)
            # TODO: implicit solvers fail because t is a tracer
        
        self.forward = jax.jit(forward)
        self.forward = jax.vmap(forward, in_axes=(None, 0, 0))

        def forward_g(params, z, u):
            t = z[-1]
            z = z[:-1]

            x = z[0:self.num_differential_vars]
            y = z[self.num_differential_vars::]
            g = self.dae.g

            # params['u_func'] = u
            return g(x, y, t, params)
        
        self.forward_g = jax.jit(forward_g)
        self.forward_g = jax.vmap(forward_g, in_axes=(None, 0, 0))
        self.init_params = init_params
        self.init_state = init_state