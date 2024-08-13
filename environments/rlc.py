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

class RLC(Environment):
    def __init__(self, dt=0.01, random_seed=42, R=1, L=1, C=1, name: str = 'RLC'):
        super().__init__(dt=dt, random_seed=random_seed, name=name)

        self.R = R
        self.L = L
        self.C = C

        self.config = {
            'dt': dt,
            'R': R,
            'L': L,
            'C': C,
        }

    def _update_config(self):
        self.config['R'] = self.R
        self.config['L'] = self.L
        self.config['C'] = self.C

    def _define_dynamics(self):
        def CapacitorPE(state):
            Q = state[0]
            return 0.5 * (Q**2 / self.config['C'])
        
        def InductorPE(state):
            flux = state[1]
            return 0.5 * (flux**2 / self.config['L'])
        
        def Dissipated(state):
            jv = state[1] / self.config['L']
            return -(jv**2) * self.config['R']
        
        def H(state):
            return CapacitorPE(state) + InductorPE(state) # + Dissipated(state)
        
        def dynamics_function(state, t, control_input, jax_key):            
            '''
                if state = [q, phi]
                \dot{q} = phi / L = I
                \dot{phi} = V - phi / L * R - q / C

                J = [[0, 1],
                     [-1, 0]]
                
                r = [[0], [R]]

                control = [[0], [V]]

                dH = [[Q/C], [phi/L]]
            '''

            dH = jax.grad(H)(state)
            J = jnp.array([[0, 1],
                           [-1, 0]])
            R = jnp.array([[0, 0], [0, self.config['R']]])
            # control_input is only V, so augment control with I=0 to be augmented_control=[[I], [V]]
            augmented_control = jnp.concatenate((jnp.array([0]), control_input))
            return jnp.matmul(J - R, dH) + augmented_control
        
        def get_power(state, control_input):
            pass
        
        self.CapacitorPE = jax.jit(CapacitorPE)
        self.InductorPE = jax.jit(InductorPE)
        self.Dissipated = jax.jit(Dissipated)
        self.H = jax.jit(H)

        self.dynamics_function = (dynamics_function)
        self.get_power = jax.jit(get_power)
    
    def plot_trajectory(self, trajectory, fontsize=15, linewidth=3):
        fig = plt.figure(figsize=(5,5))

        T = np.arange(trajectory.shape[0]) * self._dt
        Q1 = trajectory[:, 0]
        Phi1 = trajectory[:, 1]

        ax = fig.add_subplot(111)
        ax.plot(T, Q1, label='Q1')
        ax.plot(T, Phi1, label='Phi1')
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()
        ax.legend()

        plt.show()


    def plot_energy(self, trajectory, fontsize=15, linewidth=3):
        fig = plt.figure(figsize=(7,4))

        T = np.arange(trajectory.shape[0]) * self._dt

        control = jax.vmap(self.control_policy, in_axes=(0,0,None))(trajectory, T, jax.random.key(0))

        CapacitorPE = jax.vmap(self.CapacitorPE, in_axes=(0,))(trajectory)
        InductorPE = jax.vmap(self.InductorPE, in_axes=(0,))(trajectory)
        Dissipated = jax.vmap(self.Dissipated, in_axes=(0,))(trajectory)
        H = jax.vmap(self.H, in_axes=(0,))(trajectory)
        Source = (trajectory[:,1] / self.L) * control[:,1] # IV
        H += Source
        ax = fig.add_subplot(111)

        ax.plot(T, CapacitorPE, color='red', label='Capacitor PE')
        ax.plot(T, InductorPE, color='blue', label='Inductor PE')
        ax.plot(T, Dissipated, color='yellow', label='Dissipated')
        ax.plot(T, Source, color='green', label='Source')
        ax.plot(T, H, color='black', label='Total Energy')

        ax.set_ylabel('$Energy$ $[J]$', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()
        ax.legend()

        plt.show()

def generate_dataset(args, env_seed: int = 501):
    save_dir = os.path.join(os.curdir, f'results/RLC_data')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    if args.type == 'train':
        # V_range = jnp.array([1.0, 2.0])
        V_range = jnp.array([0.9, 1.1]) # TODO: Try with new dataset!
        seed = env_seed
        R_range = (1.0, 1.0)
        L_range = (1.0, 1.0)
        C_range = (1.0, 1.0)
    elif args.type == 'val':
        V_range = jnp.array([0.8, 0.9])
        seed = env_seed + 1
        # C_range = (1.5, 2.0)
        # Cp_range = (1.5, 2.0)
        # L_range = (1.5, 2.0)
        R_range = (1.0, 1.0)
        L_range = (1.0, 1.0)
        C_range = (1.0, 1.0)

    params = {
        'dt': 0.01,
        'R': 1.0,
        'L': 1.0,
        'C': 1.0,
    }

    env = RLC(**params, random_seed=env_seed)
   
    dataset = None
    key = jax.random.key(seed)
    key, ckey, lkey, rkey, vkey = jax.random.split(key, 5)
    Rs = jax.random.uniform(rkey, shape=(args.n,), minval=R_range[0], maxval=R_range[1])
    Ls = jax.random.uniform(lkey, shape=(args.n,), minval=L_range[0], maxval=L_range[1])
    Cs = jax.random.uniform(ckey, shape=(args.n,), minval=C_range[0], maxval=C_range[1])
    Vs = jax.random.uniform(vkey, shape=(args.n,), minval=V_range[0], maxval=V_range[1])
    for i in tqdm(range(args.n)):
        def control_policy(state, t, jax_key):
            return jnp.array([Vs[i]])

        env.set_control_policy(control_policy)
    
        env.C = Cs[i]
        env.R = Rs[i]
        env.L = Ls[i]
        env._update_config()
        env._define_dynamics()
        x0_init_lb = jnp.array([0.0, 0.0]) # was q0 = Vs[i]
        x0_init_ub = jnp.array([0.0, 0.0]) # was q0 = Vs[i]
        new_dataset = env.gen_dataset(trajectory_num_steps=args.steps,
                                      num_trajectories=1,
                                      x0_init_lb=x0_init_lb,
                                      x0_init_ub=x0_init_ub)
        if dataset is not None:
            dataset = merge_datasets(dataset, new_dataset, params=('R', 'L', 'C'))
        else:
            dataset = new_dataset

    save_path = os.path.join(os.path.abspath(save_dir),  
        strftime(f'{args.type}_{args.n}_{args.steps}_constant_params.pkl'))
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)

    traj = dataset['state_trajectories'][-1, :, :]
    env.plot_trajectory(traj)
    env.plot_energy(traj)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)

    args =  parser.parse_args()

    generate_dataset(args)    