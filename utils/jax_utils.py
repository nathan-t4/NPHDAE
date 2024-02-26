from jax import vmap, tree_map
import jax.numpy as jnp
""" 
    Source: https://github.com/bhchiang/rt/blob/master/utils/transforms.py#L5 
    Also see: https://github.com/google/jax/discussions/5322 
"""
def pytrees_stack(pytrees, axis=0):
    results = tree_map(
        lambda *values: jnp.stack(values, axis=axis), *pytrees)
    return results

def pytrees_vmap(fn):
    def g(pytrees):
        stacked = pytrees_stack(pytrees)
        results = vmap(fn)(stacked)
        return results
    return g