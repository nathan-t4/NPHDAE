import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
from environments.random_env import PHDAEEnvironment

"""
    A simple circuit that exhibits chaotic behavior. 

    https://en.wikipedia.org/wiki/Chua%27s_circuit

    The current across the nonlinear resistor R2 (the Chua diode) is:
    I2(V2) = m1 V2 + 0.5 (m0 - m1) (|V2 + 1| - |V2 - 1|)
"""

if __name__ == '__main__':
    AR = jnp.array([[-1, 0],
                    [1, -1]])
    AL = jnp.array([[-1], [0]])
    AC = jnp.array([[-1, 0],
                    [0, -1]])
    AV = jnp.array([[0], [0]])
    AI = jnp.array([[0], [0]])

    R1 = 1.0
    m0 = -8/7
    m1 = -5/7
    alpha = 15.6
    beta = 28
    C = 1 / alpha
    L = 1 / beta

    def grad_H_func(phi, jax_key, params):
        return phi / L

    def r_func(delta_V, jax_key, params):
        I2 = lambda dV : m1 * dV + 0.5 * (m0 - m1) * (jnp.abs(dV + 1) - jnp.abs(dV - 1))
        return jnp.array([delta_V[0] / R1, I2(delta_V[1])])
    
    def q_func(delta_V, jax_key, params):
        return C * delta_V
    
    def u_func(t, jax_key, params):
        return jnp.array([])
    
    dt = 1e-2
    seed = 42
    
    env = PHDAEEnvironment(AC, AR, AL, AV, AI, grad_H_func, q_func, r_func, u_func, dt, seed)
    
    curdir = os.path.abspath(os.path.curdir)
    save_dir = os.path.abspath(os.path.join(curdir, 'dgu_triangle_data'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset = env.gen_dataset(
        z0_init_lb=jnp.array([-0.5, -0.5 -0.5, 0.0, 0.0]),
        z0_init_ub=jnp.array([0.5, 0.5, 0.5, 0.0, 0.0]),
        trajectory_num_steps=500,
        num_trajectories=1,
        save_str=save_dir,
    )

    traj = dataset['state_trajectories'][0,:,:]
    T = jnp.arange(traj.shape[0]) * env.dt
    fig, ax = plt.subplots(3, 1, figsize=(15,25))
    ax[0,0].plot(T, traj[:,:,0:2], label=['q1', 'q2'])
    ax[1,0].plot(T, traj[:,:,2], label='phi')
    ax[2,0].plot(T, traj[:,:,3:5], label=['e1', 'e2', 'e3', 'e4', 'e5'])

    plt.show()


