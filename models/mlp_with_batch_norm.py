# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A minimal interface mlp module."""

from collections.abc import Iterable
from typing import Callable, Optional

import haiku as hk
import jax


# If you are forking replace this with `import haiku as hk`.
# pylint: disable=invalid-name
Module = hk.Module
initializers = hk.initializers
get_parameter = hk.get_parameter
PRNGSequence = hk.PRNGSequence
Linear = hk.Linear
dropout = hk.dropout
BatchNorm = hk.BatchNorm
# pylint: enable=invalid-name


class MLP_BM(hk.Module):
  """A multi-layer perceptron module."""

  def __init__(
      self,
      output_sizes: Iterable[int],
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      with_bias: bool = True,
      activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
      activate_final: bool = False,
      name: Optional[str] = None,
  ):
    """Constructs an MLP.

    Args:
      output_sizes: Sequence of layer sizes.
      w_init: Initializer for :class:`~haiku.Linear` weights.
      b_init: Initializer for :class:`~haiku.Linear` bias. Must be ``None`` if
        ``with_bias=False``.
      with_bias: Whether or not to apply a bias in each layer.
      activation: Activation function to apply between :class:`~haiku.Linear`
        layers. Defaults to ReLU.
      activate_final: Whether or not to activate the final layer of the MLP.
      name: Optional name for this module.

    Raises:
      ValueError: If ``with_bias`` is ``False`` and ``b_init`` is not ``None``.
    """
    if not with_bias and b_init is not None:
      raise ValueError("When with_bias=False b_init must not be set.")
    
    # bn_config = dict(bn_config)
    bn_config = dict()
    bn_config.setdefault("create_scale", True)
    bn_config.setdefault("create_offset", True)
    bn_config.setdefault("decay_rate", 0.999)
    # bn_config.setdefault("axis", (0))

    super().__init__(name=name)
    self.with_bias = with_bias
    self.w_init = w_init
    self.b_init = b_init
    self.activation = activation
    self.activate_final = activate_final
    layers = []
    output_sizes = tuple(output_sizes)
    for index, output_size in enumerate(output_sizes):
      if index != len(output_sizes) - 1:
        linear = hk.Linear(output_size=output_size,
                              w_init=w_init,
                              b_init=b_init,
                              with_bias=with_bias,
                              name="linear_%d" % index)
        bn = hk.BatchNorm(name="batchnorm_%d" % index, **bn_config)
        layers.append((linear, bn))
      else:
        linear = hk.Linear(output_size=output_size,
                              w_init=w_init,
                              b_init=b_init,
                              with_bias=with_bias,
                              name="linear_%d" % index)
        layers.append((linear,))
    
    self.layers = tuple(layers)
    self.output_size = output_sizes[-1] if output_sizes else None

  def __call__(
      self,
      inputs: jax.Array,
      is_training: bool,
      dropout_rate: Optional[float] = None,
      rng=None,
  ) -> jax.Array:
    """Connects the module to some inputs.

    Args:
      inputs: A Tensor of shape ``[batch_size, input_size]``.
      dropout_rate: Optional dropout rate.
      rng: Optional RNG key. Require when using dropout.

    Returns:
      The output of the model of size ``[batch_size, output_size]``.
    """
    if dropout_rate is not None and rng is None:
      raise ValueError("When using dropout an rng key must be passed.")
    elif dropout_rate is None and rng is not None:
      raise ValueError("RNG should only be passed when using dropout.")

    rng = hk.PRNGSequence(rng) if rng is not None else None
    num_layers = len(self.layers)
    out = inputs
    for i, layer in enumerate(self.layers):
      out = layer[0](out)
      if i < (num_layers - 1) or self.activate_final:
        # Batch normalization if training
        out = layer[1](out, is_training=is_training)
        # Only perform dropout if we are activating the output.
        if dropout_rate is not None:
          out = hk.dropout(next(rng), dropout_rate, out)
        out = self.activation(out)

    return out