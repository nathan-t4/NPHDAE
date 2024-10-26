from abc import abstractmethod
import optax

class OptimizerFactory:
    def __init__(self, optimizer_setup):
        self.optimizer_setup = optimizer_setup

    @abstractmethod
    def get_optimizer(self):
        """
        Get the optimizer object.
        To be implemented by child classes.
        """

class AdamFactory(OptimizerFactory):
    def get_optimizer(self):        
        if 'clipping' not in self.optimizer_setup:
            return optax.adam(self.optimizer_setup['learning_rate'])
        else:
            return optax.chain(
                optax.clip(self.optimizer_setup['clipping']),
                optax.adam(self.optimizer_setup['learning_rate']),
            )          

class AdamWFactory(OptimizerFactory):
    def get_optimizer(self):
        if 'weight_decay' not in self.optimizer_setup:
            opt = optax.adamw(self.optimizer_setup['learning_rate'])
        else:
            opt = optax.adamw(self.optimizer_setup['learning_rate'],
                              weight_decay=self.optimizer_setup['weight_decay'])
        
        if 'clipping' not in self.optimizer_setup:
            return opt
        else:
            return optax.chain(
                optax.clip(self.optimizer_setup['clipping']),
                opt
            )

optimizer_factories = {
    'adam': AdamFactory,
    'adamw': AdamWFactory,
}

def get_optimizer(optimizer_setup):
    optimizer_factory = optimizer_factories[optimizer_setup['name']](optimizer_setup)
    return optimizer_factory.get_optimizer()