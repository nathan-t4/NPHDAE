import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import diffrax


def force(t, args):
    return 0.5 * jnp.cos(t)


def vector_field(t, y, args):
    return jnp.array([[0, 1], [0, 0]]) @ y + jnp.array([[0],[force(t, args)]])


@jax.jit
def solve(y0, args):
    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Euler()
    t0 = 0
    t1 = 10
    dt0 = 0.1
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 1000))
    sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, args=args, saveat=saveat)
    return sol


y0 = jnp.array([[0],[0]])
args = ()
sol = solve(y0, args)
print(sol.ys.shape, sol.ys[30:35])