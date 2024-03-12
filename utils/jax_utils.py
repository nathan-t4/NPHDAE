import jax
import jax.numpy as jnp

from functools import partial
""" 
    Source: https://github.com/bhchiang/rt/blob/master/utils/transforms.py#L5 
    Also see: https://github.com/google/jax/discussions/5322 
"""
def pytrees_stack(pytrees, axis=0):
    results = jax.tree_map(
        lambda *values: jnp.stack(values, axis=axis), *pytrees)
    return results

def pytrees_vmap(fn):
    def g(pytrees):
        stacked = pytrees_stack(pytrees)
        results = jax.vmap(fn)(stacked)
        return results
    return g

def scan_loop(f, carry, *xs):
    """ 
        Source: https://github.com/google/jax/discussions/10401
        TODO: fix 
    """
    # base case takes n+1 arguments representing slices (s, x1[i], x2[j], ...)
    # loop = lambda s, *x_is: f(s, *x_is)

    # induction case: scan over one argument, eliminating it
    def scan_one_more(loop, x):
        def new_loop(s, *xs):
            s, _ = jax.lax.scan(lambda s, x_i : loop, s, x) # s is state
            return s
        return new_loop

    loop = f
    # compose
    for x in reversed(xs):
        loop = scan_one_more(f, x)
    
    return loop(carry, *xs)

def double_scan(f, init, x1s, x2s):
    def outer_loop(carry, x_i):
        carry, ys = jax.lax.scan(lambda s, x1 : f(s, x1, x_i), carry, x1s)
        return carry, ys

    return jax.lax.scan(outer_loop, init, x2s) # inner loop

def num_parameters(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))