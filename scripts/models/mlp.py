import flax.linen as nn
from typing import Sequence
from scripts.models.activation_fns import *

class MLP(nn.Module):
    feature_sizes: Sequence[int]
    activation: str = 'swish'
    dropout_rate: float = 0
    deterministic: bool = True
    with_layer_norm: bool = False

    @nn.compact
    def __call__(self, input, training: bool=False):
        x = input
        if self.activation == 'swish':
            activation_fn = nn.swish
        elif self.activation == 'relu':
            activation_fn = nn.relu
        elif self.activation == 'sin':
            activation_fn = SineLayer()
        elif self.activation == 'softplus':
            activation_fn = nn.softplus
        elif self.activation == 'squareplus':
            activation_fn = squareplus

        for i, size in enumerate(self.feature_sizes):
            x = nn.Dense(features=size)(x)
            if i != len(self.feature_sizes) - 1:
                x = activation_fn(x)
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        if self.with_layer_norm:
            x = nn.LayerNorm()(x)
        return x
    
class vmapMLP(nn.Module):
    feature_sizes: Sequence[int]
    activation: str = 'swish'
    dropout_rate: float = 0
    deterministic: bool = True
    with_layer_norm: bool = False
    @nn.compact
    def __call__(self, xs):
        vmapMLP = nn.vmap(MLP, variable_axes={'params': None}, split_rngs={'params': False}, in_axes=0) 
        return vmapMLP(feature_sizes=self.feature_sizes,
                       activation=self.activation,
                       dropout_rate=self.dropout_rate,
                       deterministic=self.deterministic,
                       with_layer_norm=self.with_layer_norm,
                       name='MLP')(xs)
