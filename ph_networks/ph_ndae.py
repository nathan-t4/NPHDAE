import jax

import haiku as hk

import sys
sys.path.append('../')
from ph_dae import PHDAE

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
        self.rng_key = rng_key
        self.init_rng_key = rng_key

    def _build_ph_ndae(self):

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
            pass

        self.forward = jax.jit(forward)

    def predict_trajectory(self, params, z0):
        pass