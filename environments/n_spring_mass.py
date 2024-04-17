import jax

from datetime import datetime

import matplotlib.pyplot as plt

import numpy as np

import pickle
import os
from functools import partial
from copy import deepcopy

import jax
import jax.numpy as jnp

from environments.environment import Environment
from environments.utils import merge_datasets

###### Code to generate a dataset of double-pendulum trajectories ######

class NMassSpring(Environment):
    """
    Object representing a damped mass spring system.

    Parameters
    ----------
    dt :
        The timestep used to simulate the system dynamics.
    random_seed : 
        Manually set the random seed used to generate initial states.
    m :
        The masses [kg].
    k : 
        The spring constants [N/m].
    y :
        The unstretched length of springs [m].
    b :
        The damping coefficients [Ns/m].
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
                m: jnp.ndarray = jnp.array([1, 1]),
                k: jnp.ndarray = jnp.array([1, 1]),
                b: jnp.ndarray = jnp.array([0, 0]),
                y: jnp.ndarray = jnp.array([1, 1]),
                state_measure_spring_elongation : bool =True,
                nonlinear_damping : bool = False,
                nonlinear_spring : bool = False,
                name : str = 'N_Spring_Mass'
                ):
        """
        Initialize the n spring-mass environment object.
        """

        super().__init__(dt=dt, random_seed=random_seed, name=name)
        
        assert (len(m) == len(k) == len(b)), 'parameter arrays should have the same length'
        self.m = m
        self.k = k
        self.b = b
        self.N = int(len(m))
        self.y = jnp.ones(self.N)

        self.colors = plt.colormaps['Dark2'](np.linspace(0.15, 0.85, self.N))

        self.state_measure_spring_elongation = state_measure_spring_elongation
        self.nonlinear_damping = nonlinear_damping
        self.nonlinear_spring = nonlinear_spring

        self.config = {
            'dt' : dt,
            'm': m,
            'k': k,
            'b': b,
            'y': y,
            'state_measure_spring_elongation' : state_measure_spring_elongation,
            'nonlinear_damping' : nonlinear_damping,
            'nonlinear_spring' : nonlinear_spring,
            'name' : name,
        }

    def update_config(self):
        self.config['dt'] = self.config['dt']
        self.config['m'] = self.m
        self.config['k'] = self.k
        self.config['y'] = self.y
        self.config['b'] = self.b
        self.config['state_measure_spring_elongation'] = self.state_measure_spring_elongation
        self.config['nonlinear_damping'] = self.nonlinear_damping
        self.config['nonlinear_spring'] = self.nonlinear_spring

    def _define_dynamics(self):

        def PE(state):
            """
            The system's potential energy.
            """
            qs = state[::2]
            if self.state_measure_spring_elongation:
                if self.nonlinear_spring:
                    return sum([jnp.cosh(q) - 1 for q in qs]) 
                else:
                    return sum([1/2 * k * q**2 for k, q in zip(self.k, qs)]) 
            else:
                PE = 0
                if self.nonlinear_spring:
                    for i, (q, y, k) in enumerate(zip(qs, self.y, self.k)):
                        if i == 0:
                            PE += jnp.cosh(q - y) - 1
                        else:
                            PE += jnp.cosh((qs[i] - qs[i - 1]) - y) - 1
                else:
                    for i, (q, y, k) in enumerate(zip(qs, self.y, self.k)):
                        if i == 0:
                            PE += 1/2 * k * (q - y)**2
                        else:
                            PE += 1/2 * k * ((qs[i] - qs[i - 1]) - y)**2            
                return PE

        def KE(state):
            """
            The system's kinetic energy.
            """
            ps = state[1::2]
            return sum([p**2 / (2 * m) for (p, m) in zip(ps, self.m)])

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
            J = []
            if self.state_measure_spring_elongation:
                # TODO
                J = jnp.array([[0.0, 1.0, 0.0, 0.0],
                               [-1.0, 0.0, 1.0, 0.0],
                               [0.0, -1.0, 0.0, 1.0],
                               [0.0, 0.0, -1.0, 0.0]])
            else:
                for _ in range(self.N):
                    J.append([[0.0, 1.0],
                              [-1.0, 0.0]])
            J = jax.scipy.linalg.block_diag(*jnp.array(J))

            if self.nonlinear_damping:
                ps = state[1::2]
                dampings = self.b * ps**2 / self.m**2
            else:
                dampings = self.b

            diagonals = [0] * (2 * self.N)
            diagonals[1::2] = dampings
            R = jnp.diag(jnp.array(diagonals))

            g = []
            for _ in range(self.N):
                g.extend([0.0, 1.0])
            g = jnp.array(g)

            return jnp.matmul(J - R, dh) + g * control_input

        def get_power(x, u):
            """
            Get the power of the various components of the port-Hamiltonian system.
            """
            dh = jax.grad(H)(x)
            J = []
            if self.state_measure_spring_elongation:
                # TODO
                J = jnp.array([[0.0, 1.0, 0.0, 0.0],
                                [-1.0, 0.0, 1.0, 0.0],
                                [0.0, -1.0, 0.0, 1.0],
                                [0.0, 0.0, -1.0, 0.0]])
            else:
                for _ in range(self.N):
                    J.append([[0.0, 1.0],
                              [-1.0, 0.0]])
            J = jax.scipy.linalg.block_diag(jnp.array(J))

            if self.nonlinear_damping:
                ps = x[1::2]
                dampings = self.b * ps**2 / self.m**2
            else:
                dampings = self.b

            diagonals = [0] * (2 * self.N)
            diagonals[1::2] = dampings
            R = jnp.diag(jnp.array(diagonals))

            g = []
            for _ in range(self.N):
                g.extend([0.0, 1.0])
            g = jnp.array(g)

            J_pow = jnp.matmul(dh.transpose(), jnp.matmul(J, dh))
            R_pow = jnp.matmul(dh.transpose(), jnp.matmul(- R, dh))
            g_pow = jnp.matmul(dh.transpose(), g * u)

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
        qs = trajectory[:, ::2]
        if self.state_measure_spring_elongation:
            # TODO: generalize
            y1 = self.y[0]
            y2 = self.y[1]
            q1 = trajectory[:, 0] + y1 * jnp.ones(trajectory[:,0].shape)# np.zeros((2*len(dampings),))
            q2 = trajectory[:, 2] + q1 + y2 * jnp.ones(trajectory[:,2].shape)

        ax = fig.add_subplot(211)
        for i, q in enumerate(qs.T):
            ax.plot(T, q, linewidth=linewidth, color=self.colors[i], label=f'q{i+1}')
        ax.set_ylabel(r'$q$ $[m]$', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()
        ax.legend()

        # p1 = trajectory[:, 1]
        # p2 = trajectory[:, 3]
        ps = trajectory[:,1::2]
        ax = fig.add_subplot(212)
        for i, p in enumerate(ps.T):
            ax.plot(T, p, linewidth=linewidth, color=self.colors[i], label=f'p{i}')
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

    """
    N = args.N

    m = [1.0, 1.0]
    k = [1.2, 1.5]
    b = [1.7, 1.5]
    if N == 3:
        m.append(1.0)
        k.append(1.5)
        b.append(1.6)
    elif N == 4:
        m.extend([1.0, 1.0])
        k.extend([1.5, 1.4])
        b.extend([1.6, 1.5])
    elif N == 5:
        m.extend([1.0, 1.0, 1.0])
        k.extend([1.5, 1.4, 1.7])
        b.extend([1.6, 1.5, 1.2])

    """ Set damping constant to zero """
    b = [0] * N

    params = {
        'dt': 0.01,
        'm': jnp.array(m),
        'k': jnp.array(k),
        'b': jnp.array(b),
        'state_measure_spring_elongation': False,
        'nonlinear_damping': False, # True
        'nonlinear_spring': False,
    }

    def control_policy(state, t, jax_key, aux_data):
        # TODO: Generalize to inputs on all masses. Right now there is only input on the last mass
        control_name = args.control.lower()
        if control_name == 'random':
            return 1.0 * jax.random.uniform(jax_key, shape=(1,), minval=-1.0, maxval=1.0)
        elif control_name == 'one_random_continuous':
            coefficients, scales, offsets = aux_data
            f = lambda x : jnp.sum(coefficients * jnp.cos(scales * x + offsets), axis=1)
            control = [0] * (2 * N)
            control[-1] = f(t).squeeze()[-1]
            return jnp.array(control)
        elif control_name == 'all_random_continuous':
            coefficients, scales, offsets = aux_data
            f = lambda x : jnp.sum(coefficients * jnp.cos(scales * x + offsets), axis=1)
            control = [0] * (2 * N)
            control[1::2] = f(t).squeeze()
            return jnp.array(control)
        
        elif control_name == 'passive':
            return jnp.zeros(2*N)
        else:
            raise RuntimeError('Invalid control flag')

    curdir = os.path.abspath(os.path.curdir)
    save_dir = os.path.abspath(os.path.join(curdir, f'results/{N}_mass_spring_data'))

    t = time.time()

    rng = np.random.default_rng(env_seed)

    # TODO: are these initial conditions sensible?
    # x0_init_lb = jnp.array([-.05] * 2 * N)
    # x0_init_ub = jnp.array([.0] * 2 * N)

    means = deepcopy(params)
    means['control_coef'] = jnp.zeros((N,1))
    means['control_scale'] = jnp.zeros((N,1))
    means['control_offset'] = jnp.zeros((N,1))

    train_scales = {
        'm': 0.1,
        'k': 0.1,
        # 'b': 0.1,
        'b': 0.0,
        'control_coef': 0.5,
        'control_scale': 0.5,
        'control_offset': 0.5,
    }
    val_scales = {
        'm': 0.1,
        'k': 0.1,
        # 'b': 0.1,
        'b': 0.0,
        'control_coef': 0.5,
        'control_scale': 0.5,
        'control_offset': 0.5,
    }

    key = jax.random.key(env_seed)

    env = None
    if args.type == 'training':
        env = NMassSpring(**params, random_seed=501)
    
        y = sum([jnp.concatenate((jnp.zeros((i)), jnp.array(env.y[i:]))) for i in range(len(env.y))])

        x0_init_lb = [0] * 2 * N
        x0_init_lb[::2] = 0.5 * y
        x0_init_lb = jnp.array(x0_init_lb)

        x0_init_ub = [0] * 2 * N
        x0_init_ub[::2] = 1.5 * y
        x0_init_ub = jnp.array(x0_init_ub)


        dataset = None
        for _ in range(args.n_train):
            def rng_param(key, means_key, scales_key, shape=()):
                N = len(means[means_key])
                train_range = []
                for i in range(N):
                    range_one = rng.uniform(means[means_key][i] - train_scales[scales_key], 
                                            means[means_key][i] + train_scales[scales_key],
                                            size=shape)
                    train_range.append(range_one)
                return train_range
                # return jax.random.uniform(rng_key, shape, minval=mean-scale, maxval=mean+scale)
            
            env.m = jnp.array(rng_param(key, 'm', 'm'))
            env.k = jnp.array(rng_param(key, 'k', 'k'))
            env.b = jnp.array(rng_param(key, 'b', 'b'))

            coefficients = jnp.array(rng_param(key, 'control_coef', 'control_coef', (3,)))
            scales       = jnp.array(rng_param(key, 'control_scale', 'control_scale', (3,)))
            offsets      = jnp.array(rng_param(key, 'control_offset', 'control_offset', (3,)))

            aux_data = (coefficients, scales, offsets)
            env.update_config()
            env.set_control_policy(partial(control_policy, aux_data=aux_data))    
            
            new_dataset = env.gen_dataset(trajectory_num_steps=1500, 
                                          num_trajectories=1,
                                          x0_init_lb=x0_init_lb,
                                          x0_init_ub=x0_init_ub)
            if dataset is not None:
                dataset = merge_datasets(dataset, new_dataset, params=('m', 'k', 'b'))
            else:
                dataset = new_dataset

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(os.path.abspath(save_dir),  
            datetime.now().strftime(
                f'train_{args.n_train}_{train_scales["m"]}_{train_scales["control_coef"]}_{args.control.lower()}.pkl'))
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)

    elif args.type == 'validation': 
        env = NMassSpring(**params, random_seed=501)

        y = sum([jnp.concatenate((jnp.zeros((i)), env.y[i:])) for i in range(len(env.y))])

        x0_init_lb = [0] * 2 * N
        x0_init_lb[::2] = 0.5 * y
        x0_init_lb = jnp.array(x0_init_lb)

        x0_init_ub = [0] * 2 * N
        x0_init_ub[::2] = 1.5 * y
        x0_init_ub = jnp.array(x0_init_ub)

        dataset = None
        for i in range(args.n_val):
            def get_validation_range(jax_key, means_key, scales_key, shape=()):
                assert means_key in means.keys()
                assert scales_key in train_scales.keys() and scales_key in val_scales.keys()
                N = len(means[means_key])
                sampler = lambda a, b, shape: rng.choice(np.array([a, b]), shape)
                validation_range = []
                for i in range(N):
                    jax_key, rng_key = jax.random.split(jax_key)
                    range_one = rng.uniform(means[means_key][i] + train_scales[scales_key], 
                                            means[means_key][i] + train_scales[scales_key] + val_scales[scales_key])
                    range_two = rng.uniform(means[means_key][i] - train_scales[scales_key] - val_scales[scales_key], 
                                            means[means_key][i] - train_scales[scales_key])
                    validation_range.append(sampler(range_one, range_two, shape))
                return validation_range
            keys = jax.random.split(key, 7)
            env.m        = jnp.array(get_validation_range(keys[1], 'm', 'm'))
            env.k        = jnp.array(get_validation_range(keys[2], 'k', 'k'))
            env.b        = jnp.array(get_validation_range(keys[3], 'b', 'b'))
            coefficients = jnp.array(get_validation_range(keys[4], 'control_coef', 'control_coef', (3,)))
            scales       = jnp.array(get_validation_range(keys[5], 'control_scale', 'control_scale', (3,)))
            offsets      = jnp.array(get_validation_range(keys[6], 'control_offset', 'control_offset', (3,)))
            
            env.update_config()
            aux_data = (coefficients, scales, offsets)

            env.set_control_policy(partial(control_policy, aux_data=aux_data))    

            new_dataset = env.gen_dataset(trajectory_num_steps=1500, 
                                          num_trajectories=1,
                                          x0_init_lb=x0_init_lb,
                                          x0_init_ub=x0_init_ub)
            if dataset is not None:
                dataset = merge_datasets(dataset, new_dataset, params=('m', 'k', 'b'))
            else:
                dataset = new_dataset

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(
            os.path.abspath(save_dir),  
            datetime.now().strftime(
                f"val_{args.n_val}_{val_scales['m']}_{val_scales['control_coef']}_{args.control.lower()}.pkl")
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
    parser.add_argument('--N', type=int, default=2)
    parser.add_argument('--type', type=str, default='training')
    parser.add_argument('--control', type=str, default='passive')
    parser.add_argument('--n_train', type=int, default=1500)
    parser.add_argument('--n_val', type=int, default=20)
    args = parser.parse_args()

    assert(args.type.lower() == 'training' or args.type.lower() == 'validation')

    generate_dataset(args)