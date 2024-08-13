import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.typing import Array, Dtype

@jax.jit
def squareplus(x: Array, b: Array = 4) -> Array:
    r"""Squareplus activation function.

    From Flax source code

    Computes the element-wise function

    .. math::
        \mathrm{squareplus}(x) = \frac{x + \sqrt{x^2 + b}}{2}

    as described in https://arxiv.org/abs/2112.11687.

    Args:
        x : input array
        b : smoothness parameter
    """
    x = jnp.asarray(x)
    b = jnp.asarray(b)
    y = x + jnp.sqrt(jnp.square(x) + b)
    return y / 2

class SineLayer(nn.Module):
    # TODO: for testing purposes
    param_dtype: Dtype = jnp.float32
    omega_init: float = 30

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        omega = self.param('omega', lambda k : jnp.asarray(self.omega_init, self.param_dtype))
        layer = nn.Dense(features=inputs.shape[-1], param_dtype=self.param_dtype)
        return jnp.sin(omega * layer(inputs))