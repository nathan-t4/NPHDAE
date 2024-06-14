import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial

def circuit_one_dynamics(state, t, control_input, params):
    C, C_prime, L = params
    J = jnp.array([[0, 1, 0],
                   [-1, 0, 1],
                   [0, -1, 0]])
    z = jnp.array([state[0] / C, state[1] / L, state[2] / C_prime])
    g = jnp.array([[0, 0, 0], [0, 0, 0], [0, 0, -1]])
    return jnp.matmul(J, z) + jnp.dot(g, control_input)

def circuit_two_dynamics(state, t, control_input, params):
    C, C_prime, L = params
    J = jnp.array([[0, 1], [-1, 0]])
    z = jnp.array([state[0] / C, state[1] / L])
    g = jnp.array([[0, 0], [0, -1]])
    return jnp.matmul(J, z) + jnp.matmul(g, control_input)

def coupled_circuit_dynamics(state, t, control_input, params):
    C, C_prime, L = params
    J = jnp.array([[0, 1, 0, 0, 0],
                   [-1, 0, 1, 0, 0],
                   [0, -1, 0, 0, -1],
                   [0, 0, 0, 0, 1],
                   [0, 0, 1, -1, 0]])
    z = jnp.array([state[0] / C, state[1] / L, state[2] / C_prime, state[3] / C, state[4] / L])
    return jnp.matmul(J, z)

def circuit_1_hamiltonian(state1, params):
    C, C_prime, L = params
    return 0.5 * (state1[0] ** 2 / C + state1[1] ** 2 / L + state1[2] ** 2 / C_prime)

def circuit_2_hamiltonian(state2, params):
    C, C_prime, L = params
    return 0.5 * (state2[0] ** 2 / C + state2[1] ** 2 / L)


def interconnected_circuit_dynamics(state, t, control_input, params):
    C, C_prime, L = params
    J1 = jnp.array([[0, 1, 0],
                    [-1, 0, 1],
                    [0, -1, 0]])
    J2 = jnp.array([[0, 1], 
                    [-1, 0]])
    Jc = jnp.array([[0, 0], [0, 0], [0, -1]])
    J = jnp.block([[J1, Jc], [-Jc.T, J2]])
    # state = jnp.array([state1, state2])
    # z = jnp.array([z1, z2])
    # z = jnp.array([state[0] / C, state[1] / L, state[2] / C_prime, state[3] / C, state[4] / L])
    state1 = state[:3]
    state2 = state[3:]
    dH1 = jax.grad(partial(circuit_1_hamiltonian, params=params))(state1)
    dH2 = jax.grad(partial(circuit_2_hamiltonian, params=params))(state2)
    z = jnp.concatenate((dH1, dH2))
    return jnp.matmul(J, z)

def euler(f, x, t, dt):
    return x + f(x, t) * dt

def simulate(x_dot, x0, dt, num_steps, controller, integrator):
    ts = jnp.linspace(
    0, (num_steps + 1), num=num_steps + 1, endpoint=False,  dtype=jnp.int32
    )
    xs = [x0]
    control_inputs = []
    for t in ts:
        cur_state = xs[-1]
        control = controller(cur_state, t)
        next_state = integrator(partial(x_dot, control_input=control), cur_state, t, dt)
        control_inputs.append(control)
        xs.append(next_state)

    # Append the last control input again to make the trajectory length the same.
    control_inputs.append(control)

    xs = jnp.array(xs[:-1])
    control_inputs = jnp.array(control_inputs[:-1])

    return ts, xs, control_inputs

C = 1.0
C_prime = 0.5
L = 1.0
params = (C, C_prime, L)
dt = 0.01
num_steps = 100

no_control = lambda x, t : jnp.zeros(len(x))

x10 = jnp.array([0.9, 0, 1.0])
x1_dot = partial(circuit_one_dynamics, params=params)
ts1, xs1, u1 = simulate(x1_dot, x10, dt, num_steps, no_control, euler)

x20 = jnp.array([0.8, 0])
x2_dot = partial(circuit_two_dynamics, params=params)
ts2, xs2, u2 = simulate(x2_dot, x20, dt, num_steps, no_control, euler)

xc0 = jnp.array([0.9, 0, 1.0, 0.8, 0])
xc_dot = partial(coupled_circuit_dynamics, params=params)
xc_dot = jax.jit(xc_dot)
tsc, xsc, _ = simulate(xc_dot, xc0, dt, num_steps, no_control, euler)

xc_hat_dot = partial(interconnected_circuit_dynamics, params=params)
tsc_hat, xsc_hat, _ = simulate(xc_hat_dot, jnp.concatenate((x10, x20)), dt, num_steps, no_control, euler)
# print(xs1.shape, xs2.shape, xsc.shape)
# Phi2 = xs2[:,1]
# Q3 = xs1[:,2]

# def circuit_one_control_policy(state, t):
#     return jnp.array([0, 0, Phi2[t] / L])

# ts1m, xs1m, u1m = simulate(x1_dot, x10, dt, num_steps, circuit_one_control_policy, euler)

# def circuit_two_control_policy(state, t):
#     return jnp.array([0, -Q3[t] / C_prime])

# ts2m, xs2m, u2m = simulate(x2_dot, x20, dt, num_steps, circuit_two_control_policy, euler)

axes = plt.figure(layout="constrained", figsize=(15,10)).subplot_mosaic(
    """
    CD
    EE
    """
)

# axes['A'].plot(ts1, xs1, label=['q1', 'phi1', 'q3'])
# axes['A'].legend()

# axes['B'].plot(ts2, xs2, label=['q2', 'phi2'])
# axes['B'].legend()

axes['C'].plot(tsc, xsc, label=['q1', 'phi1', 'q3', 'q2', 'phi2'])
axes['C'].legend()

axes['D'].plot(tsc_hat, xsc_hat, label=['q1', 'phi1', 'q3', 'q2', 'phi2'])
axes['D'].legend()

axes['E'].plot(tsc_hat, xsc_hat - xsc, label=['q1 error', 'phi1 error', 'q3 error', 'q2 error', 'phi2 error'])
axes['E'].legend()

plt.show()