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

class LC(Environment):
    def __init__(self, dt=0.01, random_seed=42, C=1, L=1, name: str = 'LC'):
        super().__init__(dt=dt, random_seed=random_seed, name=name)

        self.C = C
        self.L = L

        self.config = {
            'dt': dt,
            'C': C,
            'L': L,
        }

        self.omega = np.sqrt((L * C) ** (-1))

    def _define_dynamics(self):
        def CapacitorPE(state):
            Q = state[0]
            return 0.5 * (Q**2 / self.C)
        
        def InductorPE(state):
            flux = state[1]
            return 0.5 * (flux**2 / self.L)
        
        def H(state):
            return CapacitorPE(state) + InductorPE(state)
        
        def dynamics_function(state, t, control_input, jax_key):
            dH = jax.grad(H)(state)

            J = jnp.array([[0, 1],
                        [-1, 0]])
            
            R = jnp.zeros((2,2))
            
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
        Q = trajectory[:, 0]
        Phi = trajectory[:, 1]

        ax = fig.add_subplot(211)
        ax.plot(T, Q)
        ax.set_ylabel('$Q$', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()
        ax.legend()

        ax = fig.add_subplot(212)
        ax.plot(T, Phi)
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

class CLC(Environment):
    def __init__(self, dt=0.01, random_seed=42, C=1, C_prime=1, L=1, name: str = 'LC'):
        super().__init__(dt=dt, random_seed=random_seed, name=name)

        self.C = C
        self.C_prime = C_prime
        self.L = L

        self.config = {
            'dt': dt,
            'C': C,
            'L': L,
        }

        self.omega = np.sqrt((L * C) ** (-1))

    def _define_dynamics(self):
        def CapacitorPE(state):
            Q = state[0]
            return 0.5 * (Q**2 / self.C + Q**2 / self.C_prime)
        
        def InductorPE(state):
            flux = state[1]
            return 0.5 * (flux**2 / self.L)
        
        def H(state):
            return CapacitorPE(state) + InductorPE(state)
        
        def dynamics_function(state, t, control_input, jax_key):
            dH = jax.grad(H)(state)

            J = jnp.array([[0, 1],
                           [-1, 0]])
            
            R = jnp.zeros((2,2))
            
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
        Q = trajectory[:, 0]
        Phi = trajectory[:, 1]

        ax = fig.add_subplot(211)
        ax.plot(T, Q)
        ax.set_ylabel('$Q$', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()
        ax.legend()

        ax = fig.add_subplot(212)
        ax.plot(T, Phi)
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
    if args.circuit == 'lc':
        circuit = LC
        params = {
            'dt': 0.01,
            'C': 1,
            'L': 1,
        }
    
    elif args.circuit == 'clc':
        circuit = CLC
        params = {
                'dt': 0.01,
                'C': 1,
                'C_prime': 0.5,
                'L': 1,
        }

    if args.type == 'train':
        x0_init_lb = jnp.array([0.0, 0.0])
        x0_init_ub = jnp.array([2.0, 0.0])
    elif args.type == 'val':
        x0_init_lb = jnp.array([2.0, 0.0])
        x0_init_ub = jnp.array([2.5, 0.0])

    env = circuit(**params, random_seed=env_seed)
    dataset = None

    for _ in tqdm(range(args.n)):
        new_dataset = env.gen_dataset(trajectory_num_steps=args.steps,
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
    parser.add_argument('--circuit', type=str, required=True)
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)

    args = parser.parse_args()

    generate_dataset(args)    
 