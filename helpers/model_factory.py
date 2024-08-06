
from abc import abstractmethod

import sys
sys.path.append('..')

import jax

class ModelFactory():
    """Abstract factory that creates a machine learning model."""

    def __init__(self, model_setup) -> None:
        self.model_setup = model_setup.copy()

    @abstractmethod
    def create_model(self, rng_key : jax.random.PRNGKey):
        """
        Instantiate the model object from the model setup parameters.
        To be implemented by child classes.
        """

class PHNDAE_Factory(ModelFactory):
    """Factory that creates a vanilla neural ODE."""

    def create_model(self, rng_key : jax.random.PRNGKey):
        """Instantiate a vanilla neural ODE."""
        from models.ph_ndae import PHNDAE
        return PHNDAE(rng_key=rng_key,
                    model_setup=self.model_setup)
    
class MlpFactory(ModelFactory):
    """Factory that creates a multi-layer perceptron."""

    def create_model(self, rng_key : jax.random.PRNGKey):
        from models.mlp import MLP
        """Instantiate a multi-layer perceptron."""
        return MLP(rng_key=rng_key,
                model_setup=self.model_setup)


class TimeDependentNodeFactory(ModelFactory):
    """Factory that creates a vanilla time-dependent neural ODE."""

    def create_model(self, rng_key : jax.random.PRNGKey):
        """Instantiate a vanilla time-dependent neural ODE."""
        from models.time_dependent_neural_ode import TimeDependentNODE
        return TimeDependentNODE(rng_key=rng_key,
                    model_setup=self.model_setup)

model_factories = {
    'phndae' : PHNDAE_Factory,
    'mlp' : MlpFactory,
    'time_dependent_node' : TimeDependentNodeFactory,
}

def get_model_factory(model_setup):
    factory_name = model_setup['model_type']
    return model_factories[factory_name](model_setup)