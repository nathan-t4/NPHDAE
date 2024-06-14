import os
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import strftime
from functools import partial
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

        ax = fig.add_subplot(212)
        ax.plot(T, Phi)
        ax.set_ylabel('Flux', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()

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

class LC1(Environment):
    def __init__(self, dt=0.01, random_seed=42, C=1, C_prime=1, L=1, name: str = 'LC1'):
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

    def _update_config(self):
        self.config['C'] = self.C
        self.config['C_prime'] = self.C_prime
        self.config['L'] = self.L
        self.omega = np.sqrt((self.L * self.C) ** (-1))
        self.alpha = self.C / self.C_prime

    def _define_dynamics(self):
        def CapacitorPE(state):
            Q1 = state[0]
            Q3 = state[2]
            return 0.5 * (Q1**2 / self.config['C'] + Q3**2 / self.config['C_prime'])
        
        def InductorPE(state):
            flux = state[1]
            return 0.5 * (flux**2 / self.config['L'])
        
        def H(state):
            return CapacitorPE(state) + InductorPE(state)
        
        def dynamics_function(state, t, control_input, jax_key):
            # dH = jax.grad(H)(state)

            # J = jnp.array([[0, 1],
            #                [-1, 0]])
            
            # R = jnp.zeros((2,2))

            # g = jnp.array([[1, 0], [0, 0]])
            # control_input = jnp.array([control_input, 0])
            # return jnp.matmul(J - R, dH) + jnp.matmul(g, control_input) # x_dot
            z = jnp.array([state[0] / self.config['C'], 
                           state[1] / self.config['L'], 
                           state[2] / self.config['C_prime']])
            J = jnp.array([[0, 1, 0],
                           [-1, 0, 1],
                           [0, -1, 0]])
            g = jnp.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, -1]])
            
            return jnp.matmul(J, z) + jnp.matmul(g, control_input)
            
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
        Phi = trajectory[:, 1]
        Q3 = trajectory[:, 2]

        ax = fig.add_subplot(311)
        ax.plot(T, Q1)
        ax.set_ylabel('$Q1$', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()

        ax = fig.add_subplot(312)
        ax.plot(T, Phi)
        ax.set_ylabel('Flux', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()

        ax = fig.add_subplot(313)
        ax.plot(T, Q3)
        ax.set_ylabel('$Q3$', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()

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

class LC2(Environment):
    def __init__(self, dt=0.01, random_seed=42, C=1, L=1, name: str = 'LC2'):
        super().__init__(dt=dt, random_seed=random_seed, name=name)
        self.C = C
        self.L = L

        self.config = {
            'dt': dt,
            'C': C,
            'L': L,
        }

        self.omega = np.sqrt((L * C) ** (-1))

    def _update_config(self):
        self.config['C'] = self.C
        self.config['L'] = self.L
        self.omega = np.sqrt((self.L * self.C) ** (-1))
    
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
            # dH = jax.grad(H)(state)

            # J = jnp.array([[0, 1],
            #                [-1, 0]])
            
            # R = jnp.zeros((2,2))

            # g = jnp.array([[1,0],
            #               [0,-1]])
            # return jnp.matmul(J - R, dH) + jnp.matmul(g, control_input) # x_dot
            z = jnp.array([state[0] / self.config['C'],
                           state[1] / self.config['L']])
            J = jnp.array([[0, 1],
                           [-1, 0]])
            g = jnp.array([[0, 0,],
                           [0, -1]])
            return jnp.matmul(J, z) + jnp.matmul(g, control_input)

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

        ax = fig.add_subplot(212)
        ax.plot(T, Phi)
        ax.set_ylabel('Flux', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()

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
    save_dir = os.path.join(os.curdir, f'results/{args.circuit.upper()}_data')

    if args.circuit.lower() == 'lc':
        circuit = LC
        params = {
            'dt': 0.01,
            'C': 1,
            'L': 1,
        }
    
    elif args.circuit.lower() == 'lc1':
        circuit = LC1
        sys_params = ('C', 'C_prime', 'L')
        params = {
                'dt': 0.01,
                'C': 1,
                'C_prime': 0.5,
                'L': 1,
        }
    elif args.circuit.lower() == 'lc2':
        circuit = LC2
        sys_params = ('C', 'L')
        params = {
            'dt': 0.01,
            'C': 1,
            'L': 1,
        }
    else:
        raise NotImplementedError()

    if args.type == 'train':
        seed = env_seed
        if args.circuit == 'lc1':
            x0_init_lb = jnp.array([0.0, 0.0, 0.0])
            x0_init_ub = jnp.array([2.0, 0.0, 1.0])
        elif args.circuit == 'lc2':
            x0_init_lb = jnp.array([0.0, 0.0])
            x0_init_ub = jnp.array([2.0, 0.0])

        control_mag = 1.0
        C_range = (0.5, 1.5)
        C_prime_range = (0.5, 1.5)
        L_range = (0.5, 1.5)

    elif args.type == 'val':
        seed = env_seed + 1
        if args.circuit == 'lc1':
            x0_init_lb = jnp.array([2.0, 0.0, 1.0]) 
            x0_init_ub = jnp.array([2.5, 0.0, 1.5])
        elif args.circuit == 'lc2':
            x0_init_lb = jnp.array([2.0, 0.0]) 
            x0_init_ub = jnp.array([2.5, 0.0])
        control_mag = 0.0
        C_range = (1.5, 2.0)
        C_prime_range = (1.5, 2.0)
        L_range = (1.5, 2.0)
        
        # C_range = (0.75, 0.75)                    
        # C_prime_range = (0.75, 0.75) 
        # L_range = (0.75, 0.75)
    
    key = jax.random.key(seed)
    env = circuit(**params, random_seed=seed)
    dataset = None

    if args.circuit == 'lc1':
        key, cpkey = jax.random.split(key)
        C_primes = jax.random.uniform(cpkey, shape=(args.n,), minval=C_prime_range[0], maxval=C_prime_range[1])

    key, ckey, lkey = jax.random.split(key, 3)
    Cs = jax.random.uniform(ckey, shape=(args.n,), minval=C_range[0], maxval=C_range[1])
    Ls = jax.random.uniform(lkey, shape=(args.n,), minval=L_range[0], maxval=L_range[1])

    for i in tqdm(range(args.n)): 
        # Update parameters
        if args.circuit.lower() == 'lc1':
            env.C_prime = C_primes[i]
        
        env.C = Cs[i]
        env.L = Ls[i]
        env._update_config()
        env._define_dynamics()    

        if args.circuit.lower() == 'lc2':
            # Train with random (constant) voltages - can be positive and negative
            key, subkey = jax.random.split(key)
            k = control_mag * jax.random.uniform(subkey, shape=(6,), minval=-1.0, maxval=1.0)
        
            def control_policy(state, t, jax_key, aux_data):
                k = aux_data
                i = k[0] * jnp.sin(k[1] * t + k[2]) 
                v = k[3] * jnp.sin(k[4] * t + k[5])
                return jnp.array([0, v])

            env.set_control_policy(partial(control_policy, aux_data=k))

        elif args.circuit.lower() == 'lc1':
            key, subkey = jax.random.split(key)   
            k = control_mag * jax.random.uniform(subkey, shape=(3,), minval=-1.0, maxval=1.0)

            def control_policy(state, t, jax_key, aux_data):
                k = aux_data
                # return jnp.array([0, 0, k[0] * jnp.sin(k[1] * t + k[2])])
                return jnp.array([0, 0, 0])
            
            env.set_control_policy(partial(control_policy, aux_data=k))

        new_dataset = env.gen_dataset(trajectory_num_steps=args.steps,
                                      num_trajectories=1,
                                      x0_init_lb=x0_init_lb,
                                      x0_init_ub=x0_init_ub)
        
        if dataset is not None:
            dataset = merge_datasets(dataset, new_dataset, params=sys_params)            
        else:
            dataset = new_dataset
                
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    save_path = os.path.join(os.path.abspath(save_dir),  
        strftime(f'{args.type}_{args.n}_{args.steps}.pkl'))
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)

    traj = dataset['state_trajectories'][-1, :, :]
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
 