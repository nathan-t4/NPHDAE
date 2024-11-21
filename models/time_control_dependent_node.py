import jax
import jax.numpy as jnp
import numpy as np

import haiku as hk
from jax.experimental.ode import odeint

from .common import get_params_struct, choose_nonlinearity

import sys
sys.path.append('../')

from helpers.model_factory import get_model_factory
from helpers.integrator_factory import integrator_factory

class TimeControlDependentNODE(object):

    def __init__(self,
                rng_key : jax.random.PRNGKey,
                model_setup : dict,):
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
        self.rng_key = rng_key
        self.init_rng_key = rng_key
        self.input_dim = model_setup['input_dim']
        self.output_dim = model_setup['output_dim']
        self.dt = model_setup['dt']

        self.model_setup = model_setup.copy()

        # Initialize the neural network ode.
        self._build_neural_ode()
        self.params_shapes, self.count, self.params_tree_struct = \
            get_params_struct(self.init_params)

    def _build_neural_ode(self):
        """ 
        This function builds a neural network to directly estimate future state 
        values. It assigns self.forward(), self.init_params, and self.vector_field().
        """

        network_setup = self.model_setup['network_setup']
        self.rng_key, subkey = jax.random.split(self.rng_key)
        network = get_model_factory(network_setup).create_model(subkey)

        init_params = network.init_params

        integrator = integrator_factory(self.model_setup['integrator'])

        def network_forward(params, x, u, t):
            input = jnp.concatenate([x, u, jnp.array([t])])
            return network.forward(params, input)

        def forward(params, x, u):

            t = x[-1]
            x = x[:-1]

            def f_approximator(x, t):
                return network_forward(params, x, u, t)

            return integrator(f_approximator, x, t, self.dt)

        forward = jax.jit(forward)
        forward = jax.vmap(forward, in_axes=(None, 0, 0))

        self.init_params = init_params
        self.forward = forward
        self.vector_field = network_forward