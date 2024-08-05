from abc import abstractmethod
import sys
sys.path.append('..')

class trainerFactory():
    """Abstract factory method that creates model trainer objects."""

    def __init__(self, trainer_setup):
        self.trainer_setup = trainer_setup.copy()

    @abstractmethod
    def create_trainer(self, model):
        """Create the model trainer object."""

class SGDTrainerFactory(trainerFactory):
    """Factory method that creates a standard SGD model trainer object."""

    def create_trainer(self, model):
        """Create a standard SGD model trainer object."""
        from trainers.sgd_trainer import SGDTrainer

        return SGDTrainer(model,
                            model.init_params,
                            self.trainer_setup)

# A mapping from the names of the trainer types to the 
# appropriate trainer factories.
trainer_factories = {
    'sgd' : SGDTrainerFactory,
}

def get_trainer_factory(trainer_setup):
    """
    Return the appropriate trainer factory, given the configuration
    information provided in the trainer_setup dictionary.
    """
    factory_name = trainer_setup['trainer_type']
    return trainer_factories[factory_name](trainer_setup)