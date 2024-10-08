import sys

sys.path.append('../')
from jax._src.random import PRNGKey as PRNGKey
import jax.numpy as jnp
import jax

from models.ph_dae_random import RandomPHDAE
import os
from datetime import datetime
import pickle
import time
import tqdm

import matplotlib.pyplot as plt
from functools import partial

class Random_DGU_PH_DAE():

    def __init__(
            self,
            AC,
            AR,
            AL,
            AV,
            AI,
            grad_H_func,
            q_func,
            r_func,
            u_func,
            params=None,
            dt=0.01,
            seed=42,
            name='DGU_DAE'
    ):
        self.dt = dt
        self.name = name
        self.params = params
        self.config = {
            'AC' : AC,
            'AR' : AR,
            'AL' : AL,
            'AV' : AV,
            'AI' : AI,
            'dt' : dt,
            'params' : params,
            'seed' : seed,
            'name' : name,
        }

        assert type(seed) is int
        self._random_seed = seed
        self._rng_key = jax.random.PRNGKey(seed)

        self.u_func = u_func
        self.dae = RandomPHDAE(AC, AR, AL, AV, AI, grad_H_func, q_func, r_func, u_func)

    def gen_random_trajectory(
            self, 
            z0_init_lb: jnp.array, 
            z0_init_ub: jnp.array,
            trajectory_num_steps: int = 50, 
            rng_key: jax.random.PRNGKey = None) -> tuple:
        """
        Generate a random trajectory.
        """

        # Randomly generate an initial state.
        shape = z0_init_lb.shape
        key, subkey, daekey = jax.random.split(rng_key, 3)
        z0val = jax.random.uniform(subkey, 
                                    shape=shape, 
                                    minval=z0_init_lb, 
                                    maxval=z0_init_ub)
        
        T = jnp.arange(0.0, step=self.dt, stop=self.dt * trajectory_num_steps)

        control_inputs = jax.vmap(self.u_func, in_axes=(0, None, None))(T, daekey, None)

        # Generate the trajectory. Note that the dae solver automatically
        # finds the closest initial state for the algebraic variables
        return self.dae.solve(z0val, T, daekey, self.params), T, control_inputs

    def gen_dataset(self,
                    z0_init_lb : jnp.array,
                    z0_init_ub : jnp.array,
                    trajectory_num_steps : int = 500,
                    num_trajectories : int = 200,
                    save_str=None):
        """
        Generate a dataset of system trajectories with 
        randomly sampled initial points.

        Parameters
        ----------
        trajectory_num_steps : 
            The number of timesteps to include in each trajectory of data.
        num_trajectories: 
            The total number of trajectories to include in the dataset.
        x0_init_lb : 
            Jax Numpy array representing the lower bound of possible initial 
            system states when generating the dataset.
        x0_init_ub :
            Jax Numpy array representing the upper bound of possible initial 
            system states when generating the dataset.
        save_str :
            A path string indicating the folder in which to save the dataset.

        Returns
        -------
        dataset :
            Dictionary containing the generated trajectory data.
        """
        dataset = {}

        # Save the size of the timestep used to simulate the data.
        dataset['config'] = self.config.copy()

        self._rng_key, subkey = jax.random.split(self._rng_key)
        trajectory, timesteps, control_inputs  = self.gen_random_trajectory(
                                                    z0_init_lb, 
                                                    z0_init_ub, 
                                                    trajectory_num_steps=\
                                                        trajectory_num_steps,
                                                    rng_key = subkey,
                                                )
        dataset['state_trajectories'] = jnp.array([trajectory])
        dataset['control_inputs'] = jnp.array([control_inputs])
        dataset['timesteps'] = jnp.array([timesteps])

        # training_dataset = jnp.array([jnp.stack((state, next_state), axis=0)])
        for traj_ind in tqdm.tqdm(range(1, num_trajectories), desc='Generating training trajectories'):
            self._rng_key, subkey = jax.random.split(self._rng_key)
            while True:
                try:
                    trajectory, timesteps, control_inputs = self.gen_random_trajectory(
                                                                z0_init_lb, 
                                                                z0_init_ub, 
                                                                trajectory_num_steps=\
                                                                    trajectory_num_steps,
                                                                rng_key=subkey,
                                                            )
                except ValueError:
                    pass
                break
            dataset['state_trajectories'] = jnp.concatenate(
                    (dataset['state_trajectories'], jnp.array([trajectory])), axis=0
                )
            dataset['control_inputs'] = jnp.concatenate(
                    (dataset['control_inputs'], jnp.array([control_inputs])), axis=0
                )
            dataset['timesteps'] = jnp.concatenate(
                    (dataset['timesteps'], jnp.array([timesteps])), axis=0
                )
        
        if save_str is not None:
            assert os.path.isdir(save_str)
            save_path = os.path.join(os.path.abspath(save_str),  
                            datetime.now().strftime(self.name + '_%Y-%m-%d-%H-%M-%S.pkl'))
            # jnp.save(save_path, dataset)
            with open(save_path, 'wb') as f:
                pickle.dump(dataset, f)

        return dataset

if __name__ == '__main__':
    AC = jnp.array([[0.0], [0.0], [1.0]])
    AR = jnp.array([[-1.0], [1.0], [0.0]])
    AL = jnp.array([[0.0], [1.0], [-1.0]])
    AV = jnp.array([[1.0], [0.0], [0.0]])
    AI = jnp.array([[0.0], [0.0], [-1.0]])

    dt = 1e-8
    R = 0.2; L = 1.8e-3; C = 2.2e-3
    I_magnitude = 1.0
    V_magnitude = 100.0
    init_charge_range = jnp.array([-1.0, 1.0]) * 1e-3
    init_flux_range = jnp.array([-1.0, 1.0]) * 1e-3

    # R = 1; L = 1; C = 1 
    # I_magnitude = 0.1
    # V_magnitude = 1.0

    def r_func(delta_V, jax_key, params=None):
        return delta_V / R
    
    def q_func(delta_V, jax_key, params=None):
        return C * delta_V
    
    def grad_H_func(phi, jax_key, params=None):
        return phi / L
    
    def u_func(t, jax_key, params):
        jax_key, ik, vk = jax.random.split(jax_key, 3)
        # i = jax.random.uniform(ik, minval=0.0, maxval=1.0)
        # v = jax.random.uniform(vk, minval=0.0, maxval=1.0)
        # return jnp.array([I_magnitude * i, V_magnitude * v])
        return jnp.array([I_magnitude, V_magnitude])
    
    env = Random_DGU_PH_DAE(AC, AR, AL, AV, AI, grad_H_func, q_func, r_func, u_func, dt=dt)

    curdir = os.path.abspath(os.path.curdir)
    # save_dir = os.path.abspath(os.path.join(curdir, 'results/DGU_data'))
    save_dir = os.path.abspath(os.path.join(curdir, 'dgu_dae_data'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    t = time.time()
    print('starting simulation')

    seed = 42 # for testing
    # seed = 41 # for training
    dataset = env.gen_dataset(
        z0_init_lb=jnp.array(
            [init_charge_range[0], init_flux_range[0], V_magnitude, V_magnitude, 0.0, 0.0]
        ),
        z0_init_ub=jnp.array(
            [init_charge_range[1], init_flux_range[1], V_magnitude, V_magnitude, V_magnitude, V_magnitude]
        ),
        trajectory_num_steps=800, # 1000 for training, 800 for testing.
        num_trajectories=20, # 500 for training, 20 for testing
        save_str=save_dir,
    )

    traj = dataset['state_trajectories'][0]
    plt.plot(traj, label=['q', 'phi', 'e1', 'e2', 'e3', 'jv'])
    plt.legend()
    plt.show()

    print(dataset['control_inputs'].shape)
    control = dataset['control_inputs'][0]
    plt.plot(control, label=['i', 'v'])
    plt.legend()
    plt.show()

    print(time.time() - t)