import jax

import haiku as hk

import sys
sys.path.append('../')
from models.ph_dae import PHDAE
import sacred

from .common import get_params_struct, choose_nonlinearity
import jax.numpy as jnp
from helpers.model_factory import get_model_factory

class PHNDAE():

    def __init__(
            self,
            rng_key : jax.random.PRNGKey,
            model_setup : dict,
        ):
        """
        Constructor for the neural ODE.

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

        self.model_setup = model_setup.copy()

        self._get_num_vars()
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

    def _build_ph_ndae(self):

        init_params = {}

        # Define the H function for the inductors
        self.rng_key, subkey = jax.random.split(self.rng_key)
        H_net = get_model_factory(self.model_setup['H_net_setup']).create_model(subkey)
        init_params['grad_H_func'] = H_net.init_params

        num_inductors = self.num_inductors
        def grad_H_func(phi, params):
            H = lambda x : jnp.sum(H_net.forward(params=params, x=x))
            phi = jnp.reshape(phi, (num_inductors, 1))
            return jax.vmap(jax.grad(H), 0)(phi).reshape((num_inductors,))

        self.H_net = H_net
        self.grad_H_func = jax.jit(grad_H_func)

        # def grad_H_func(phi, params):
        #     return phi
        # self.grad_H_func = jax.jit(grad_H_func)

        def r_func(delta_V, params=None):
            return delta_V / 1.0
        self.r_func = jax.jit(r_func)
        init_params['r_func_params'] = None
    
        def q_func(delta_V, params=None):
            return 1.0 * delta_V
        self.q_func = jax.jit(q_func)
        init_params['q_func_params'] = None
        
        def u_func(t, params=None):
            return jnp.array(self.model_setup['u_func'](t, params))
        self.u_func = jax.jit(u_func)
        init_params['u_func_params'] = None

        self.init_params = init_params

        self.dae = PHDAE(
            self.AC, 
            self.AR, 
            self.AL, 
            self.AV, 
            self.AI, 
            self.grad_H_func,
            self.q_func, 
            self.r_func, 
            self.u_func
        )

        def forward(params, z):
            t = z[-1]
            z = z[:-1]
            return self.dae.solver.solve_dae_one_timestep_rk4(z, t, self.dt, params)

        self.forward = jax.jit(forward)
        self.forward = jax.vmap(forward, in_axes=(None, 0))

    def predict_trajectory(self,
                            params,
                            initial_state : jnp.ndarray,
                            num_steps : int,
                            rng_key : jax.random.PRNGKey = jax.random.PRNGKey(0),):        
        """
        Predict the system trajectory from an initial state.
        
        Parameters
        ----------
        params :
            An instantiation of the network parameters.
        initial_state :
            An array representing the system initial state.
        num_steps : 
            Number of steps to include in trajectory.
        """
        times = jnp.arange(0, num_steps * self.dt, self.dt)

        traj = self.dae.solver.solve_dae(initial_state, times, params)

        trajectory = {
            'state_trajectory' : jnp.array(traj),
            'times' : jnp.array(times),
        }

        return trajectory