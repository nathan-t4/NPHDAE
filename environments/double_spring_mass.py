import jax

from datetime import datetime

import matplotlib.pyplot as plt

import numpy as np
# from scipy.integrate import odeint
from jax.experimental.ode import odeint

import pickle
import os
from functools import partial
from copy import deepcopy

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

    def update_config(self):
        self.config['dt'] = self.config['dt']
        self.config['m1'] = self.m1
        self.config['k1'] = self.k1
        self.config['y1'] = self.y1
        self.config['b1'] = self.b1
        self.config['m2'] = self.m2
        self.config['k2'] = self.k2
        self.config['y2'] = self.y2
        self.config['b2'] = self.b2
        self.config['state_measure_spring_elongation'] = self.state_measure_spring_elongation
        self.config['nonlinear_damping'] = self.nonlinear_damping
        self.config['nonlinear_spring'] = self.nonlinear_spring

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

    def control_policy(state, t, jax_key, aux_data):
        control_name = args.control.lower()
        if control_name == 'random':
            return 1.0 * jax.random.uniform(jax_key, shape=(1,), minval=-1.0, maxval=1.0)
        elif control_name == 'random_continuous':
            coefficients, scales, offsets = aux_data
            f = lambda x : coefficients[0] * jnp.cos(scales[0] * x + offsets[0]) + coefficients[1] * jnp.cos(scales[1] * x + offsets[1]) + coefficients[2] * jnp.cos(scales[2] * x + offsets[2])  
            return jnp.array([f(t)])
        elif control_name == 'sin':
            return 5.0 * jnp.array([jnp.sin(t)])
        elif control_name == 'passive':
            return jnp.array([0]) # zero input
        else:
            raise RuntimeError('Invalid control flag')
    
    curdir = os.path.abspath(os.path.curdir)
    save_dir = os.path.abspath(os.path.join(curdir, 'results/double_mass_spring_data'))

    t = time.time()

    rng = np.random.default_rng(env_seed)

    # Different datasets to test generalization (given fixed control policy)

    # System 1 parameters
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

    # System 2 parameters
    # params = {
    #     'dt': 0.01,
    #     'm1': 2.0,
    #     'm2': 2.0,
    #     'k1': 2.4,
    #     'k2': 3.0,
    #     'b1': 3.4,
    #     'b2': 3.0,
    #     'state_measure_spring_elongation': False,
    #     'nonlinear_damping': True,
    #     'nonlinear_spring': False,
    # }

    x0_init_lb = jnp.array([-.05, -.05, -.05, -.05])
    x0_init_ub = jnp.array([.0, .0, .0, .0])

    means = deepcopy(params)
    means['control_coef'] = 0
    means['control_scale'] = 0
    means['control_offset'] = 0

    train_scales = {
        'm': 0.1,
        'k': 0.1,
        'b': 0.1,
        'control_coef': 0.5,
        'control_scale': 0.5,
        'control_offset': 0.5,
    }
    val_scales = {
        'm': 0.5,
        'k': 0.5,
        'b': 0.5,
        'control_coef': 0.2,
        'control_scale': 0.2,
        'control_offset': 0.2,
    }

    key = jax.random.key(env_seed)

    env = None
    if args.type == 'training':
        env = DoubleMassSpring(**params, random_seed=501)
        dataset = None
        for _ in range(args.n_train):
            rng_param = lambda mean, scale, size : rng.uniform(mean - scale, mean + scale, size)
            env.m1 = rng_param(means['m1'], train_scales['m'], None)
            env.m2 = rng_param(means['m2'], train_scales['m'], None)
            env.k1 = rng_param(means['k1'], train_scales['k'], None)
            env.k2 = rng_param(means['k2'], train_scales['k'], None)
            env.b1 = rng_param(means['b1'], train_scales['b'], None)
            env.b2 = rng_param(means['b2'], train_scales['b'], None)

            rng_key, key = jax.random.split(key)
            coefficients = rng_param(means['control_coef'], train_scales['control_coef'], (3,))
            scales = rng_param(means['control_scale'], train_scales['control_scale'], (3,))
            offsets = rng_param(means['control_offset'], train_scales['control_offset'], (3,))
            aux_data = (coefficients, scales, offsets)

            env.update_config()
            env.set_control_policy(partial(control_policy, aux_data=aux_data))    
            
            new_dataset = env.gen_dataset(trajectory_num_steps=1500, 
                                          num_trajectories=1,
                                          x0_init_lb=x0_init_lb,
                                          x0_init_ub=x0_init_ub)
            if dataset is not None:
                dataset = merge_datasets(dataset, new_dataset)
            else:
                dataset = new_dataset

        assert os.path.isdir(save_dir)
        save_path = os.path.join(os.path.abspath(save_dir),  
            datetime.now().strftime(f'train_{args.n_train}_%H-%M-%S.pkl'))
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)

    elif args.type == 'validation': 
        env = DoubleMassSpring(**params, random_seed=501)
        dataset = None
        for i in range(args.n_val):
            rng_key, key = jax.random.split(key)

            def get_validation_range(jax_key, means_key, scales_key, shape=()):
                assert means_key in means.keys()
                assert scales_key in train_scales.keys() and scales_key in val_scales.keys()
                rng_key, jax_key = jax.random.split(jax_key)
                sampler = lambda a, b, shape: jax.random.choice(rng_key, jnp.array([a, b]), shape)
                range_one = rng.uniform(means[means_key] + train_scales[scales_key], 
                                        means[means_key] + train_scales[scales_key] + val_scales[scales_key])
                range_two = rng.uniform(means[means_key] - train_scales[scales_key] - val_scales[scales_key], 
                                        means[means_key] - train_scales[scales_key])
                return sampler(range_one, range_two, shape)

            env.m1          = get_validation_range(rng_key, 'm1', 'm')
            env.m2          = get_validation_range(rng_key, 'm2', 'm')
            env.k1          = get_validation_range(rng_key, 'k1', 'k')
            env.k2          = get_validation_range(rng_key, 'k2', 'k')
            env.b1          = get_validation_range(rng_key, 'b1', 'b')
            env.b2          = get_validation_range(rng_key, 'b2', 'b')
            coefficients    = get_validation_range(rng_key, 'control_coef', 'control_coef', (3,))
            scales          = get_validation_range(rng_key, 'control_scale', 'control_scale', (3,))
            offsets         = get_validation_range(rng_key, 'control_offset', 'control_offset', (3,))
            
            env.update_config()
            aux_data = (coefficients, scales, offsets)
            env.set_control_policy(partial(control_policy, aux_data=aux_data))    

            new_dataset = env.gen_dataset(trajectory_num_steps=1500, 
                                          num_trajectories=1,
                                          x0_init_lb=x0_init_lb,
                                          x0_init_ub=x0_init_ub)
            if dataset is not None:
                dataset = merge_datasets(dataset, new_dataset)
            else:
                dataset = new_dataset

        assert os.path.isdir(save_dir)
        save_path = os.path.join(
            os.path.abspath(save_dir),  
            datetime.now().strftime(
                f"val_{args.n_val}_{val_scales['m']}_{val_scales['control_coef']}.pkl")
        )
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
    parser.add_argument('--n_train', type=int, default=1500)
    parser.add_argument('--n_val', type=int, default=20)
    args = parser.parse_args()

    assert(args.type.lower() == 'training' or args.type.lower() == 'validation')

    generate_dataset(args)