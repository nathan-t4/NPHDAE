import os
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
from time import strftime
from environments.utils import *
from environments.environment import Environment

class SynchronousGenerator(Environment):
    def __init__(self, dt=0.01, random_seed=42, rm=1, rr=1, d=1, M=1, L_aa0=1, L_ab0=1, L_afd=1, L_akq=1, L_ffd=1, L_akd=1, L_kkd=1, L_kkq=1, name: str = 'SynchronousGenerator'):
        super().__init__(dt=dt, random_seed=random_seed, name=name)

        self.rm = rm              # 3-phase stator resistance
        self.rr = rr              # 3-phase rotor resistance
        self.d = d                # mechanical damping
        self.M = M                # rotational inertia of the rotor
        self.L_aa0 = L_aa0
        self.L_ab0 = L_ab0
        self.L_afd = L_afd
        self.L_akd = L_akd
        self.L_akq = L_akq
        self.L_ffd = L_ffd
        self.L_kkd = L_kkd
        self.L_kkq = L_kkq

        self.config = {
            'dt': dt,
            'rm': rm,
            'rr': rr,
            'd': d,
            'M': M,
            'L_aa0': L_aa0,
            'L_ab0': L_ab0,
            'L_afd': L_afd,
            'L_akd': L_akd,
            'L_akq': L_akq,
            'L_ffd': L_ffd,
            'L_kkd': L_kkd,
            'L_kkq': L_kkq,
        }
        
        self.Rsl = jnp.diag(jnp.array([self.rm, self.rm, self.rm]))
        self.Rr = jnp.diag(jnp.array([self.rr, self.rr, self.rr]))

    def get_inv_L(self, t):
        L_ess = jnp.array([[self.L_aa0, -self.L_ab0, -self.L_ab0],
                            [-self.L_ab0, self.L_aa0, -self.L_ab0],
                            [-self.L_ab0, -self.L_ab0, self.L_aa0]])
    
        L_ers = jnp.array([
            [self.L_afd * jnp.cos(t), self.L_akd * jnp.cos(t), -self.L_akq * jnp.sin(t)],
            [self.L_afd * jnp.cos(t - 2 * jnp.pi / 3), self.L_akd * jnp.cos(t - 2 * jnp.pi / 3), -self.L_akq * jnp.sin(t - 2 * jnp.pi / 3)],
            [self.L_afd * jnp.cos(t + 2 * jnp.pi / 3), self.L_akd * jnp.cos(t + 2 * jnp.pi / 3), -self.L_akq * jnp.sin(t + 2 * jnp.pi / 3)]
        ])

        L_err = jnp.array([[self.L_ffd, self.L_akd, 0],
                            [self.L_akd, self.L_kkd, 0],
                            [0, 0, self.L_kkq]])
        
        L = jnp.block([[L_ess, L_ers],     # Inductance matrix
                        [L_ers.T, L_err]])
                
        return jnp.linalg.inv(L)

    def _update_config(self):
        self.config['rm'] = self.rm
        self.config['rr'] = self.rr
        self.config['d'] = self.d
        self.config['M'] = self.M
        self.config['L_aa0'] = self.L_aa0
        self.config['L_ab0'] = self.L_ab0
        self.config['L_afd'] = self.L_afd
        self.config['L_akd'] = self.L_akd
        self.config['L_akq'] = self.L_akq
        self.config['L_ffd'] = self.L_ffd
        self.config['L_kkd'] = self.L_kkd
        self.config['L_kkq'] = self.L_kkq

    def _define_dynamics(self):
        def GeneratorPE(state):
            phis = state[:6]
            theta = state[7]
            return 0.5 * (phis.T @ self.get_inv_L(theta) @ phis)
        
        def RotorKE(state):
            p = state[6]
            return 0.5 * (p**2 / self.config['M'])
        
        def H(state):
            return GeneratorPE(state) + RotorKE(state)
        
        def dynamics_function(state, t, control_input, jax_key):
            dH = jax.grad(H)(state)
            zeros33 = jnp.zeros((3,3))
            zeros31 = jnp.zeros((3,1))
            zeros21 = jnp.zeros((2,1))

            J = jnp.block([[zeros33, zeros33, zeros31, zeros31],
                           [zeros33, zeros33, zeros31, zeros31],
                           [zeros31.T, zeros31.T, 0, -1],
                           [zeros31.T, zeros31.T, 1, 0]])
                
            R = jax.scipy.linalg.block_diag(self.Rsl, self.Rr, self.d, 0)

            g = jnp.block([[zeros31, zeros31],
                             [jnp.array([[1],[0],[0]]), zeros31],
                             [zeros21, jnp.array([[1],[0]])]])
                        
            return jnp.matmul(J - R, dH) + jnp.matmul(g, control_input) # x_dot
        
        def get_power(state, control_input):
            pass
        
        self.GeneratorPE = GeneratorPE
        self.RotorKE = RotorKE
        self.H = H

        self.dynamics_function = dynamics_function
        self.get_power = get_power
    
    def plot_trajectory(self, trajectory, fontsize=15, linewidth=3):
        fig = plt.figure(figsize=(5,5))

        T = np.arange(trajectory.shape[0]) * self._dt
        PhiS = trajectory[:, 0:3]
        PhiR = trajectory[:, 3:6]
        p = trajectory[:,6]
        theta = trajectory[:,7]

        ax = fig.add_subplot(311)
        ax.plot(T, PhiS, label=['sa', 'sb', 'sc'])
        ax.plot(T, PhiR, label=['ra', 'rb', 'rc'])
        ax.set_ylabel('$Flux$', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()
        ax.legend()

        ax = fig.add_subplot(312)
        ax.plot(T, p, label='p')
        ax.set_ylabel('Momentum', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()
        ax.legend()

        ax = fig.add_subplot(313)
        ax.plot(T, theta, label='theta')
        ax.set_ylabel('Theta', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()
        ax.legend()

        plt.show()


    def plot_energy(self, trajectory, fontsize=15, linewidth=3):
        fig = plt.figure(figsize=(7,4))

        T = np.arange(trajectory.shape[0]) * self._dt

        GeneratorPE = jax.vmap(self.GeneratorPE, in_axes=(0,))(trajectory)
        RotorKE = jax.vmap(self.RotorKE, in_axes=(0,))(trajectory)
        H = jax.vmap(self.H, in_axes=(0,))(trajectory)

        ax = fig.add_subplot(111)

        ax.plot(T, GeneratorPE, color='red', label='GeneratorPE')
        ax.plot(T, RotorKE, color='blue', label='RotorKE')

        ax.plot(T, H, color='green', label='Total Energy')

        ax.set_ylabel('$Energy$ $[J]$', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()
        ax.legend()

        plt.show()

def generate_dataset(args, env_seed: int = 501):
    save_dir = os.path.join(os.curdir, f'results/PowerNetwork_data')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    if args.type == 'train':
        seed = env_seed
        control_mag = 1
        x0_init_lb = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        x0_init_ub = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
    elif args.type == 'val':
        seed = env_seed + 1
        control_mag = 1
        x0_init_lb = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        x0_init_lb = jnp.array([1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2])

    params = {
        'dt': 0.01,
        'rm': 0.1,
        'rr': 0.1,
        'd': 0.1,
        'M': 0.5,
        'L_aa0': 1,
        'L_ab0': 0.9,
        'L_afd': 1,
        'L_akd': 0.8,
        'L_akq': 1,
        'L_ffd': 1,
        'L_kkd': 1,
        'L_kkq': 1,
    }

    param_names = ('rm', 'rr', 'd', 'M')

    env = SynchronousGenerator(**params, random_seed=env_seed)
   
    dataset = None
    key = jax.random.key(seed)
    # key, rmkey, rrkey, dkey, Mkey = jax.random.split(key, 5)
    # rms = jax.random.uniform(rmkey, shape=(args.n,), minval=rm_range[0], maxval=rm_range[1])
    # rrs = jax.random.uniform(rrkey, shape=(args.n,), minval=rr_range[0], maxval=rr_range[1])
    # ds = jax.random.uniform(dkey, shape=(args.n,), minval=d_range[0], maxval=d_range[1])
    # Ms = jax.random.uniform(Mkey, shape=(args.n,), minval=M_range[0], maxval=M_range[1])

    for i in tqdm(range(args.n)):
        key, subkey = jax.random.split(key)
        k = control_mag * jax.random.uniform(subkey, shape=(6,), minval=-1.0, maxval=1.0)
        
        def control_policy(state, t, jax_key, aux_data):
            k = aux_data
            E = k[0] * jnp.sin(k[1] * t + k[2]) 
            T = k[3] * jnp.sin(k[4] * t + k[5])
            return jnp.array([E, T])
    
        env.set_control_policy(partial(control_policy, aux_data=k))
        # env.rm = rms[i]
        # env.rr = rrs[i]
        # env.d = ds[i]
        # env.M = Ms[i]
        # env._update_config()
        # env._define_dynamics()
        new_dataset = env.gen_dataset(trajectory_num_steps=args.steps,
                                      num_trajectories=1,
                                      x0_init_lb=x0_init_lb,
                                      x0_init_ub=x0_init_ub)
        if dataset is not None:
            dataset = merge_datasets(dataset, new_dataset, params=param_names)
        else:
            dataset = new_dataset

    save_path = os.path.join(os.path.abspath(save_dir),  
        strftime(f'{args.type}_{args.n}_{args.steps}.pkl'))
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)

    traj = dataset['state_trajectories'][-1, :, :]
    env.plot_trajectory(traj)
    env.plot_energy(traj)


if __name__ == '__main__':
    # from argparse import ArgumentParser
    # parser = ArgumentParser()
    # parser.add_argument('--type', type=str, default='train')
    # parser.add_argument('--n', type=int, required=True)
    # parser.add_argument('--steps', type=int, required=True)

    # args =  parser.parse_args()
    import ml_collections
    args = ml_collections.ConfigDict({'type': 'train', 'n': 1, 'steps': 1500,})

    jax.config.update("jax_debug_nans", True)
    generate_dataset(args)    