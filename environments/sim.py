import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
from helpers.integrator_factory import integrator_factory

def get_data(data):
    J = data['config']['J']
    R = data['config']['R']
    B = data['config']['B']
    Q = data['config']['Q']
    P = data['config']['P']
    thetas = data['state_trajectories']['Va']
    V = data['state_trajectories']['Vm']

    # state = jnp.c_[jnp.zeros((100,2)), thetas[:,1:], V[:,2:]] # omega, theta, V
    state = jnp.c_[thetas[:,1:], V[:,2:]] # 7

    return J, R, B, Q, P, state

def dynamics_function(state, t, J, R, B, Q, P):
    def H(state):
        # omega = state[:2]
        # theta = state[2:6]
        # V = state[6:]
        theta = state[:4]
        V = state[4:]
        v = V * jnp.exp(jax.lax.complex(0.0, 1.0) * theta[1:])
        v = jnp.imag(V)
        return 0.5 * v.T @ B @ v + Q[2:].T @ jnp.log(V) # P[1:].T @ theta

    dH = jax.grad(H)(state)
    return jnp.matmul(J - R, dH)

def get_next_state(f, state, t, dt):
    next_state = integrator_factory('adam_bashforth')(f, state, t, dt, T=1)
    return next_state
if __name__ == '__main__':
    file = 'results/grid_data/data.pkl'
    data = np.load(file, allow_pickle=True)
    J, R, B, Q, P, exp_states = get_data(data)
    J = J[2:,2:]
    R = R[2:,2:]
    B = B[2:,2:]

    print('J:', J)

    T = 100
    dt = 1
    steps = int(T // dt)
    ts = jnp.linspace(0, T, steps)
    init_state = exp_states[0]
    print(init_state)
    states = [init_state]

    for t, q, p in zip(ts[:-1], Q, P):
        dyn_fun = partial(dynamics_function, J=J, R=R, B=B, Q=q, P=p)
        next_state = get_next_state(dyn_fun, states[-1], t, dt)
        states.append(next_state)
    
    plt.figure()
    plt.plot(ts, states, label='pH')
    plt.plot(ts, exp_states, label='exp')
    plt.legend()
    plt.show()