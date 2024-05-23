import os
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import strftime
from environments.utils import *
from environments.environment import Environment

class CoupledLC(Environment):
    def __init__(self, dt=0.01, random_seed=42, C=1, C_prime=1, L=1, name: str = 'CoupledLC'):
        super().__init__(dt=dt, random_seed=random_seed, name=name)

        self.C = C
        self.C_prime = C_prime
        self.L = L

        self.config = {
            'dt': dt,
            'C': C,
            'C_prime': C_prime,
            'L': L,
        }

        self.omega = np.sqrt((L * C) ** (-1))
        self.alpha = C / C_prime

    def _define_dynamics(self):
        def CapacitorPE(state):
            Q1 = state[0]
            Q2 = state[2]
            Q3 = -(Q1 + Q2)
            return 0.5 * (Q1**2 / self.C + Q2**2 / self.C + Q3**2 / self.C_prime)
        
        def InductorPE(state):
            flux1 = state[1]
            flux2 = state[3]
            return 0.5 * (flux1**2 / self.L + flux2**2 / self.L)
        
        def H(state):
            return CapacitorPE(state) + InductorPE(state)
        
        def dynamics_function(state, t, control_input, jax_key):
            dH = jax.grad(H)(state)

            J = jnp.array([[0, 1, 0, 0],
                           [-1, 0, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, -1, 0]])
                
            R = jnp.zeros((4,4))
            
            return jnp.matmul(J - R, dH) # x_dot
        
        def get_power(state, control_input):
            pass
        
        self.CapacitorPE = jax.jit(CapacitorPE)
        self.InductorPE = jax.jit(InductorPE)
        self.H = jax.jit(H)

        self.dynamics_function = jax.jit(dynamics_function)
        self.get_power = jax.jit(get_power)
    
    def plot_trajectory(self, trajectory, fontsize=15, linewidth=3):
        fig = plt.figure(figsize=(5,5))

        T = np.arange(trajectory.shape[0]) * self._dt
        Q1 = trajectory[:, 0]
        Phi1 = trajectory[:, 1]
        Q2 = trajectory[:,2]
        Phi2 = trajectory[:,3]

        ax = fig.add_subplot(211)
        ax.plot(T, Q1, label='Q1')
        ax.plot(T, Q2, label='Q2')
        ax.set_ylabel('$Q$', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()
        ax.legend()

        ax = fig.add_subplot(212)
        ax.plot(T, Phi1, label='Phi1')
        ax.plot(T, Phi2, label='Phi2')
        ax.set_ylabel('Flux', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()
        ax.legend()

        plt.show()


    def plot_energy(self, trajectory, fontsize=15, linewidth=3):
        fig = plt.figure(figsize=(7,4))

        T = np.arange(trajectory.shape[0]) * self._dt

        CapacitorPE = jax.vmap(self.CapacitorPE, in_axes=(0,))(trajectory)
        InductorPE = jax.vmap(self.InductorPE, in_axes=(0,))(trajectory)
        H = jax.vmap(self.H, in_axes=(0,))(trajectory)

        ax = fig.add_subplot(111)

        ax.plot(T, CapacitorPE, color='red', label='Capacitor PE')
        ax.plot(T, InductorPE, color='blue', label='Inductor PE')

        ax.plot(T, H, color='green', label='Total Energy')

        ax.set_ylabel('$Energy$ $[J]$', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()
        ax.legend()

        plt.show()

def generate_dataset(args, env_seed: int = 501):
    save_dir = os.path.join(os.curdir, 'results/LC_data')
    
    # state x = [Q, \Phi] (charge, flux)\ TODO: is initial flux = 0?

    if args.type == 'train':
        x0_init_lb = jnp.array([0.0, 0.0, 2.0, 0.0])
        x0_init_ub = jnp.array([2.0, 0.0, 4.0, 0.0])
    elif args.type == 'val':
        x0_init_lb = jnp.array([2.0, 0.0, 1.0, 0.0])
        x0_init_ub = jnp.array([3.0, 0.0, 2.0, 0.0])

    params = {
        'dt': 0.01,
        'C': 1,
        'C_prime': 0.5,
        'L': 1,
    }

    env = CoupledLC(**params, random_seed=env_seed)
    save_dir = os.path.join(os.curdir, f'results/{env.name}_data')
    dataset = None

    for _ in tqdm(range(args.n)):
        new_dataset = env.gen_dataset(trajectory_num_steps=1500,
                                        num_trajectories=1,
                                        x0_init_lb=x0_init_lb,
                                        x0_init_ub=x0_init_ub)
        if dataset is not None:
            dataset = merge_datasets(dataset, new_dataset)
        else:
            dataset = new_dataset

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    save_path = os.path.join(os.path.abspath(save_dir),  
        strftime(f'{args.type}_{args.n}.pkl'))
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)

    traj = dataset['state_trajectories'][0, :, :]
    env.plot_trajectory(traj)
    env.plot_energy(traj)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--n', type=int, required=True)

    args = parser.parse_args()

    generate_dataset(args)    