import jax
import jax.numpy as jnp

import haiku as hk

from models.mlp_with_batch_norm import MLP_BM
from .common import get_params_struct, get_flat_params, unflatten_params, choose_nonlinearity, choose_weight_initialization

class MLP(object):

    def __init__(self,
                rng_key : jax.random.PRNGKey,
                model_setup : dict,
                model_name : str = 'mlp',
                training : bool = True,
                ):
        """
        Constructor for the multi-layer perceptr.

        Parameters
        ----------
        rng_key : 
            A key for random initialization of the parameters of the 
            neural networks.
        model_setup :
            A dictionary containing the parameters for the network.
        model_name :
            A name for the model of interest.
        """

        self.rng_key = rng_key
        self.init_rng_key = rng_key

        self.model_name = model_name

        self.model_setup = model_setup
        self.input_dim = model_setup['input_dim']
        self.output_dim = model_setup['output_dim']
        self.nn_setup_params = model_setup['nn_setup_params']
        self.use_batch_norm = model_setup['use_batch_norm']

        self.training = training

        # Initialize the neural network ode.
        self._build_model()
        self.params_shapes, self.count, self.params_tree_struct = \
            get_params_struct(self.init_params)

    def _build_model(self):
        """ 
        This function builds a neural network to directly estimate future state 
        values. It assigns self.forward() and self.init_params.
        """
        nn_setup_params = self.nn_setup_params.copy()
        nn_setup_params['activation'] = choose_nonlinearity(nn_setup_params['activation'])
        # nn_setup_params['w_init'] = choose_weight_initialization(nn_setup_params['w_init']) # TODO

        def mlp_forward(x):
            if self.use_batch_norm:
                return MLP_BM(**nn_setup_params)(x)
            else:
                return hk.nets.MLP(**nn_setup_params)(x)
            
        self.rng_key, subkey = jax.random.split(self.rng_key)
        if self.use_batch_norm:
            mlp_forward_pure = hk.without_apply_rng(hk.transform_with_state(mlp_forward))
            init_params = mlp_forward_pure.init(rng=subkey, x=jnp.zeros((self.input_dim,)), is_training=True)
        else:
            mlp_forward_pure = hk.without_apply_rng(hk.transform(mlp_forward))
            init_params = mlp_forward_pure.init(rng=subkey, x=jnp.zeros((self.input_dim,)))

        def forward(params, x):
            if self.use_batch_norm:
                out = mlp_forward_pure.apply(params=params, x=x, is_training=True)
            else:
                out = mlp_forward_pure.apply(params=params, x=x)
            return out

        forward = jax.jit(forward)

        self.forward = forward
        self.init_params = init_params