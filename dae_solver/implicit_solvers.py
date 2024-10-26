import jax
import jax.numpy as jnp
from jax import lax
import jax.random as random
from functools import partial

def fwd_solver(f, z_init):
  """ Fixed point iteration """
  def cond_fun(carry):
    z_prev, z = carry
    return jnp.linalg.norm(z_prev - z) > 1e-5

  def body_fun(carry):
    _, z = carry
    return z, f(z)

  init_carry = (z_init, f(z_init))
  _, z_star = lax.while_loop(cond_fun, body_fun, init_carry)
  return z_star

def newton_solver(f, z_init):
  """ Newton's method """
  f_root = lambda z: f(z) - z
  g = lambda z: z - jnp.linalg.solve(jax.jacobian(f_root)(z), f_root(z))
  return fwd_solver(g, z_init)

@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def fixed_point_layer(solver, f, z0, params):
  z_star = solver(lambda z: f(z, params), z_init=z0)
  return z_star

def fixed_point_layer_fwd(solver, f, z0, params):
  z_star = fixed_point_layer(solver, f, z0, params)
  return z_star, (z_star, params)

def fixed_point_layer_bwd(solver, f, res, z_star_bar):
  z_star, params = res
  _, vjp_a = jax.vjp(lambda params: f(z_star, params), params)
  _, vjp_z = jax.vjp(lambda z: f(z, params), z_star)
  return vjp_a(solver(lambda u: vjp_z(u)[0] + z_star_bar,
                      z_init=jnp.zeros_like(z_star_bar)))

fixed_point_layer.defvjp(fixed_point_layer_fwd, fixed_point_layer_bwd)

if __name__ == '__main__':
  """ Root finding f(x) = 0
      Fixed point of f(x) - x
  """
  ndim = 10
  W = random.normal(random.PRNGKey(0), (ndim, ndim)) / jnp.sqrt(ndim)
  x = random.normal(random.PRNGKey(1), (ndim,))
  f = lambda z, W: jnp.tanh(jnp.dot(W, z)) + x
  z0 = jnp.zeros_like(x)
  g = jax.grad(lambda W: fixed_point_layer(fwd_solver, f, z0, W).sum())(W)
  print(g[0])