import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
from environments.random_env import PHDAEEnvironment

"""
    A simple resistor network in a grid configuration (typical in the power grid design of an IC). 
"""

if __name__ == '__main__':
    AR = jnp.array([[-1, -1, 0, 0, 0, 0, 0],
                    [1, 0, 0, -1, 0, 0, -1],
                    [0, 0, 0, 0, 0, -1, 1],
                    [0, 1, -1, 0, 0, 0, 0],
                    [0, 0, 1, 1, -1, 0, 0]])
    AV = jnp.array([[-1], [0], [0], [0], [0]])
    AI = jnp.array([[0], [0], [0], [0], [0]])
    AC = jnp.array([[0], [0], [0], [0], [0]])
    AL = jnp.array([[0], [0], [0], [0], [0]])

    R = 1.0
    L = 1.0
    C = 1.0
    V = 1.0

    def r_func(delta_V, jax_key, params):
        # TODO: ICs use mosfets as resistors. I-V is nonlinear (search mosfet triode)
        return delta_V / R 
    
    def grad_H_func(phi, jax_key, params):
        return phi / L 
    
    def q_func(delta_V, jax_key, params):
        return delta_V * C 
    
    def u_func(t, jax_key, params):
        return jnp.array([V])
    
    dt = 1e-2
    seed = 42

    curdir = os.path.abspath(os.path.curdir)
    save_dir = os.path.abspath(os.path.join(curdir, 'dgu_triangle_data'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    env = PHDAEEnvironment(AC, AR, AL, AV, AI, grad_H_func, q_func, r_func, u_func, dt, seed)

    dataset = env.gen_dataset(
        z0_init_lb=jnp.zeros(6),
        z0_init_ub=jnp.zeros(6),
        trajectory_num_steps=500,
        num_trajectories=1,
        save_str=save_dir,
    )

    traj = dataset['state_trajectories'][0,:,:]
    T = jnp.arange(traj.shape[0]) * env.dt
    fig, ax = plt.subplots(2, 2, figsize=(15,25))
    ax[0,0].plot(T, traj[:,:,0:5], label=['e1', 'e2', 'e3', 'e4', 'e5'])
    ax[1,0].plot(T, traj[:,:,5], label='jv')

    plt.show()

