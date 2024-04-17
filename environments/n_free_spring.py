import os
import pickle
import jax
import jax.numpy as jnp
from tqdm import tqdm
from helpers.integrator_factory import integrator_factory

save_path = os.path.join(os.curdir, 'results/free_spring/' + 'test.pkl')

dataset = {}

num_trajectories = 20
trajectory_num_steps = 1500
config = {
    'dt': 0.001,
    'm': [1.0, 1.0],
    'k': 1.0,
    'b': 0, # TODO
    'y': 1.0,
}
N = len(config['m']) # num masses

x0_lb = jnp.array([0] * 2 * N) 
x0_ub = 1.5 * jnp.array([0, 0.1, 0, config['y']])

assert x0_lb.shape == x0_ub.shape

timesteps = jnp.arange(trajectory_num_steps) * config['dt']

f = jnp.array([[0, 1, 0, 0],
               [-config['k'], 0, config['k'], 0],
               [0, 0, 0, 1],
               [config['k'], 0, -config['k'], 0]])
b = jnp.array([[0], [-config['k'] * config['y']], [0], [config['k'] * config['y']]])

control = lambda x, t : jnp.zeros((2*N, 1))

dynamics = lambda x, t : f @ x + b + control(x, t)

dynamics = jax.jit(dynamics)

@jax.jit
def step(x, t):
    u_next = control(x, t)
    x_next = integrator(dynamics, x, timesteps, config['dt'])
    
    return x_next, u_next

integrator = integrator_factory('rk4')

trajectory = []
control_inputs = []

key = jax.random.key(0)
for i in tqdm(range(num_trajectories), desc='Generating data'): 
    states = []
    controls = []
    key, rng_key = jax.random.split(key)
    x0 = jax.random.uniform(rng_key, 
                            shape=x0_lb.shape, 
                            minval=x0_lb, 
                            maxval=x0_ub)
    x_prev = x0
    for t in timesteps:
        x_next, u_next = step(x_prev, t)
        states.append(x_next)
        controls.append(u_next)

    trajectory.append(states)
    control_inputs.append(controls)
    
print(jnp.array(trajectory)[0])
print(jnp.array(trajectory).shape) # TODO: why is this shape weird?

print(jnp.array(control_inputs))
print(jnp.array(control_inputs).shape)

print(jnp.array(timesteps))
print(jnp.array(timesteps).shape)


dataset['state_trajectories'] = jnp.array(trajectory)
dataset['control_inputs'] = jnp.array(control_inputs)
dataset['timesteps'] = jnp.array(timesteps)

print(dataset['state_trajectories'])
print(dataset['state_trajectories'].shape)

with open(save_path, 'wb') as f:
    pickle.dump(dataset, f)
