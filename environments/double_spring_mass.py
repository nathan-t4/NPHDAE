import jax

from datetime import datetime

import matplotlib.pyplot as plt

import numpy as np
# from scipy.integrate import odeint
from jax.experimental.ode import odeint

import pickle
import os
from functools import partial

import jax
import jax.numpy as jnp

from environments.environment import Environment
from environments.utils import merge_datasets

###### Code to generate a dataset of double-pendulum trajectories ######

class DoubleMassSpring(Environment):
    """
    Object representing a damped mass spring system.

    Parameters
    ----------
    dt :
        The timestep used to simulate the system dynamics.
    random_seed : 
        Manually set the random seed used to generate initial states.
    m1 :
        The mass of mass 1 [kg].
    k1 : 
        The spring constant of spring 1 [N/m].
    y1 :
        The unstretched length of spring 1 [m].
    b1 :
        The damping coefficient on mass 1 [Ns/m].
    m2 :
        The mass of mass 2 [kg].
    k2 : 
        The spring constant of spring 2 [N/m].
    y2 :
        The unstretched length of spring 2 [m].
    b2 :
        The damping coefficient on mass 2 [Ns/m].
    state_measure_spring_elongation : bool
        If True, the state of the system is measured as the elongation of the springs.
    nonlinear_damping : bool
        If True, the damping force is given by c \dot{q}^3 .
    name : 
        The name of the environment.
    """

    def __init__(self, 
                dt=0.01, 
                random_seed=42,
                m1 : jnp.float32 = 1, 
                k1 : jnp.float32 = 1, 
                y1 : jnp.float32 = 1,
                b1 : jnp.float32 = 0.0,
                m2 : jnp.float32 = 1,
                k2 : jnp.float32 = 1,
                y2 : jnp.float32 = 1,
                b2 : jnp.float32 = 0.0,
                state_measure_spring_elongation : bool =True,
                nonlinear_damping : bool = False,
                nonlinear_spring : bool = False,
                name : str = 'Double_Spring_Mass'
                ):
        """
        Initialize the double-pendulum environment object.
        """

        super().__init__(dt=dt, random_seed=random_seed, name=name)
        
        self.m1 = m1
        self.k1 = k1
        self.y1 = y1
        self.b1 = b1

        self.m2 = m2
        self.k2 = k2
        self.y2 = y2
        self.b2 = b2

        self.state_measure_spring_elongation = state_measure_spring_elongation
        self.nonlinear_damping = nonlinear_damping
        self.nonlinear_spring = nonlinear_spring

        self.config = {
            'dt' : dt,
            'm1' : m1,
            'k1' : k1,
            'y1' : y1,
            'b1' : b1,
            'm2' : m2,
            'k2' : k2,
            'y2' : y2,
            'b2' : b2,
            'state_measure_spring_elongation' : state_measure_spring_elongation,
            'nonlinear_damping' : nonlinear_damping,
            'nonlinear_spring' : nonlinear_spring,
            'name' : name,
        }

    def _define_dynamics(self):

        def PE(state):
            """
            The system's potential energy.
            """
            q1 = state[0]
            q2 = state[2]
            if self.state_measure_spring_elongation:
                if self.nonlinear_spring:
                    return (jnp.cosh(q1) - 1) + (jnp.cosh(q2) - 1)
                else:
                    return 1/2 * self.k1 * q1**2 + 1/2 * self.k2 * q2**2
            else:
                if self.nonlinear_spring:
                    return (jnp.cosh(q1 - self.y1) - 1) + (jnp.cosh((q2 - q1) - self.y2) - 1)
                else:
                    return 1/2 * self.k1 * (q1 - self.y1)**2 + 1/2 * self.k2 * ((q2 - q1) - self.y2)**2

        def KE(state):
            """
            The system's kinetic energy.
            """
            p1 = state[1]
            p2 = state[3]
            return p1**2 / (2 * self.m1) + p2**2 / (2 * self.m2)

        def H(state):
            """
            The system's Hamiltonian.
            """
            return KE(state) + PE(state)

        def dynamics_function(state : jnp.ndarray, 
                                    t: jnp.float32,
                                    control_input : jnp.ndarray = jnp.array([0.0]),
                                    jax_key : jax.random.PRNGKey = None,
                                    ) -> jnp.ndarray:
            """
            The system dynamics formulated using Hamiltonian mechanics.
            """ 
            dh = jax.grad(H)(state)

            if self.state_measure_spring_elongation:
                J = jnp.array([[0.0, 1.0, 0.0, 0.0],
                                [-1.0, 0.0, 1.0, 0.0],
                                [0.0, -1.0, 0.0, 1.0],
                                [0.0, 0.0, -1.0, 0.0]])
            else:
                J = jnp.array([[0.0, 1.0, 0.0, 0.0],
                                [-1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                                [0.0, 0.0, -1.0, 0.0]])

            if self.nonlinear_damping:
                p1 = state[1]
                p2 = state[3]
                damping1 = self.b1 * p1**2 / self.m1**2
                damping2 = self.b2 * p2**2 / self.m2**2
            else:
                damping1 = self.b1
                damping2 = self.b2
            R = jnp.array([[0.0, 0.0, 0.0, 0.0],
                        [0.0, damping1, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, damping2]])

            g = jnp.array([[0.0, 0.0, 0.0, 1.0]]).transpose()

            return jnp.matmul(J - R, dh) + jnp.matmul(g, control_input)

        def get_power(x, u):
            """
            Get the power of the various components of the port-Hamiltonian system.
            """
            dh = jax.grad(H)(x)

            if self.state_measure_spring_elongation:
                J = jnp.array([[0.0, 1.0, 0.0, 0.0],
                                [-1.0, 0.0, 1.0, 0.0],
                                [0.0, -1.0, 0.0, 1.0],
                                [0.0, 0.0, -1.0, 0.0]])
            else:
                J = jnp.array([[0.0, 1.0, 0.0, 0.0],
                                [-1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                                [0.0, 0.0, -1.0, 0.0]])

            if self.nonlinear_damping:
                p1 = x[1]
                p2 = x[3]
                damping1 = self.b1 * p1**2 / self.m1**2
                damping2 = self.b2 * p2**2 / self.m2**2
            else:
                damping1 = self.b1
                damping2 = self.b2
            R = jnp.array([[0.0, 0.0, 0.0, 0.0],
                        [0.0, damping1, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, damping2]])
            g = jnp.array([[0.0, 0.0, 0.0, 1.0]]).transpose()

            J_pow = jnp.matmul(dh.transpose(), jnp.matmul(J, dh))
            R_pow = jnp.matmul(dh.transpose(), jnp.matmul(- R, dh))
            g_pow = jnp.matmul(dh.transpose(), jnp.matmul(g, u))

            dh_dt = J_pow + R_pow + g_pow

            return dh_dt, J_pow, R_pow, g_pow

        # def dynamics_function(state : jnp.ndarray, 
        #                 t: jnp.float32,
        #                 control_input : jnp.ndarray,
        #                 jax_key : jax.random.PRNGKey = None,
        #                 ) -> jnp.ndarray:
        #     """ 
        #     Full known dynamics
        #     """
        #     q1 = state[0]
        #     p1 = state[1]
        #     q2 = state[2]
        #     p2 = state[3]
        #     if self.state_measure_spring_elongation:
        #         q1_dot = p1 / self.m1
        #         q2_dot = p2 / self.m2 - p1 / self.m1
        #         p1_dot = - self.k1 * q1 + self.k2 * q2
        #         p2_dot = - self.k2 * q2 + control_input[0]
        #     else:
        #         q1_dot = p1 / self.m1
        #         q2_dot = p2 / self.m2
        #         p1_dot = - (self.k1 * (q1 - self.y1) + self.k2 * (q1 + self.y2 - q2))
        #         p2_dot = - (self.k2 * (q2 - q1 - self.y2)) + control_input[0]
        #     return jnp.stack([q1_dot, p1_dot, q2_dot, p2_dot])

        self.PE = jax.jit(PE)
        self.KE = jax.jit(KE)
        self.H = jax.jit(H)
        self.dynamics_function = jax.jit(dynamics_function)
        self.get_power = jax.jit(get_power)

    def plot_trajectory(self, trajectory, fontsize=15, linewidth=3):
        """
        Plot a particular trajectory.
        """
        fig = plt.figure(figsize=(5,5))

        T = np.arange(trajectory.shape[0]) * self._dt

        # We want to plot the positions of the masses, not the elongations of the springs
        if self.state_measure_spring_elongation:
            q1 = trajectory[:, 0] + self.y1 * jnp.ones(trajectory[:,0].shape)
            q2 = trajectory[:, 2] + q1 + self.y2 * jnp.ones(trajectory[:,2].shape)
        else:
            q1 = trajectory[:, 0]
            q2 = trajectory[:, 2]

        ax = fig.add_subplot(211)
        ax.plot(T, q1, linewidth=linewidth, color='blue', label='q1')
        ax.plot(T, q2, linewidth=linewidth, color='red', label='q2')
        ax.set_ylabel(r'$q$ $[m]$', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()
        ax.legend()

        p1 = trajectory[:, 1]
        p2 = trajectory[:, 3]
        ax = fig.add_subplot(212)
        ax.plot(T, p1, linewidth=linewidth, color='blue', label='p1')
        ax.plot(T, p2, linewidth=linewidth, color='red', label='p2')
        ax.set_ylabel(r'$p$ $[kg\frac{m}{s}]$', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()

        plt.show()

    def plot_energy(self, trajectory, fontsize=15, linewidth=3):
        """
        Plot the kinetic, potential, and total energy of the system
        as a function of time during a particular trajectory.
        """
        fig = plt.figure(figsize=(7,4))

        T = np.arange(trajectory.shape[0]) * self._dt

        KE = jax.vmap(self.KE, in_axes=(0,))(trajectory)
        PE = jax.vmap(self.PE, in_axes=(0,))(trajectory)
        H = jax.vmap(self.H, in_axes=(0,))(trajectory)

        ax = fig.add_subplot(111)
        ax.plot(T, KE, color='red', linewidth=linewidth, label='Kinetic Energy')
        ax.plot(T, PE, color='blue', linewidth=linewidth, label='Potential Energy')
        ax.plot(T, H, color='green', linewidth=linewidth, label='Total Energy')

        ax.set_ylabel(r'$Energy$ $[J]$', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()
        ax.legend()

        plt.show()

def generate_dataset(args, env_seed: int = 501):
    """
        TODO: 
            use params dict to initialize env
            fix hardcoded parts (dataset_type == ...)
    """
    def control_policy(state, t, jax_key):
        control_name = args.control.lower()
        if control_name == 'uniform':
            return 5.0 * jax.random.uniform(jax_key, shape=(1,), minval = -1.0, maxval=1.0)
        elif control_name == 'sin':
            return 5.0 * jnp.array([jnp.sin(t)])
        elif control_name == 'passive':
            return jnp.array([t-t]) # zero input
        else:
            raise RuntimeError('Invalid control flag')
    
    curdir = os.path.abspath(os.path.curdir)
    save_dir = os.path.abspath(os.path.join(curdir, 'results/double_mass_spring_data'))

    t = time.time()

    rng = np.random.default_rng(env_seed)

    # Different datasets to test generalization (given fixed control policy)
    params = {
        'dt': 0.01,
        'm1': 1.0,
        'm2': 1.0,
        'k1': 1.2,
        'k2': 1.5,
        'b1': 1.7,
        'b2': 1.5,
        'state_measure_spring_elongation': False,
        'nonlinear_damping': True,
        'nonlinear_spring': False,
    }

    x0_init_lb = jnp.array([-.05, -.05, -.05, -.05])
    x0_init_ub = jnp.array([.0, .0, .0, .0])
    # x0_init_ub = x0_init_lb


    val_params = {}
    for k,v in params.items():
        if isinstance(v, float) and k != 'dt':
            val_params[k] = v + 0.1 * rng.uniform(-1, 1)
        # if k == 'm1' or k == 'm2':
        #     val_params[k] = v + 0.1 * rng.uniform(-1, 1)
        else:
            val_params[k] = v

    env = None
    if args.type == 'training':
        env = DoubleMassSpring(**params, random_seed=501)
        env.set_control_policy(control_policy)    
        dataset = None
        for _ in range(50):
            env.m1 = rng.uniform(0.98, 1.02)
            env.m2 = rng.uniform(0.98, 1.02)
            env.k1 = rng.uniform(1.18, 1.22)
            env.k2 = rng.uniform(1.48, 1.52)
            env.b1 = rng.uniform(1.68, 1.72)
            env.b2 = rng.uniform(1.48, 1.52)
            
            new_dataset = env.gen_dataset(trajectory_num_steps=1500, 
                                          num_trajectories=10,
                                          x0_init_lb=x0_init_lb,
                                          x0_init_ub=x0_init_ub)
            if dataset is not None:
                dataset = merge_datasets(dataset, new_dataset)
            else:
                dataset = new_dataset

        assert os.path.isdir(save_dir)
        save_path = os.path.join(os.path.abspath(save_dir),  
            datetime.now().strftime(f'train_{500}_%H-%M-%S.pkl'))
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)

    elif args.type == 'validation': 
        num_val_trajectories = 20

        env = DoubleMassSpring(**val_params, random_seed=501)
        env.set_control_policy(control_policy)

        # x0_init_lb = (rng.random()) * x0_init_lb
        # x0_init_ub = (rng.random()) * x0_init_ub
        dataset = None
        for _ in range(10):
            env.m1 = rng.uniform(1.05, 1.08)
            env.m2 = rng.uniform(1.05, 1.08)
            print(f'masses {env.m1}, {env.m2}')
            new_dataset = env.gen_dataset(trajectory_num_steps=1500, 
                                          num_trajectories=num_val_trajectories // 10,
                                          x0_init_lb=x0_init_lb,
                                          x0_init_ub=x0_init_ub)
            if dataset is not None:
                dataset = merge_datasets(dataset, new_dataset)
            else:
                dataset = new_dataset

        assert os.path.isdir(save_dir)
        save_path = os.path.join(os.path.abspath(save_dir),  
            datetime.now().strftime(f'val_{num_val_trajectories}_%H-%M-%S.pkl'))
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
    else:
        raise NotImplementedError

    print(time.time() - t)
    print(dataset.keys())
    traj = dataset['state_trajectories'][0, :, :]
    env.plot_trajectory(traj)
    env.plot_energy(traj)

if __name__ == "__main__":
    import time
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--type', type=str, default='training')
    parser.add_argument('--control', type=str, default='passive')
    args = parser.parse_args()

    assert(args.type.lower() == 'training' or args.type.lower() == 'validation')

    generate_dataset(args)