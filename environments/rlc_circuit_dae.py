import sys

sys.path.append('../')
from jax._src.random import PRNGKey as PRNGKey
import jax.numpy as jnp
import jax

from models.ph_dae import PHDAE
import os
from datetime import datetime
import pickle
import time
import tqdm

class RLC_PH_DAE():

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
            name='RLC_DAE'
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

        self.dae = PHDAE(AC, AR, AL, AV, AI, grad_H_func, q_func, r_func, u_func)

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
        key, subkey = jax.random.split(rng_key)
        z0val = jax.random.uniform(subkey, 
                                    shape=shape, 
                                    minval=z0_init_lb, 
                                    maxval=z0_init_ub)
        
        T = jnp.arange(0.0, step=self.dt, stop=self.dt * trajectory_num_steps)

        # Generate the trajectory. Note that the dae solver automatically
        # finds the closest initial state for the algebraic variables
        return self.dae.solve(z0val, T, self.params), T

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
        trajectory, timesteps = self.gen_random_trajectory(
                                        z0_init_lb, 
                                        z0_init_ub, 
                                        trajectory_num_steps=\
                                            trajectory_num_steps,
                                        rng_key = subkey,
                                    )
        dataset['state_trajectories'] = jnp.array([trajectory])
        dataset['timesteps'] = jnp.array([timesteps])

        # training_dataset = jnp.array([jnp.stack((state, next_state), axis=0)])
        for traj_ind in tqdm.tqdm(range(1, num_trajectories), desc='Generating training trajectories'):
            self._rng_key, subkey = jax.random.split(self._rng_key)
            trajectory, timesteps = self.gen_random_trajectory(
                                            z0_init_lb, 
                                            z0_init_ub, 
                                            trajectory_num_steps=\
                                                trajectory_num_steps,
                                            rng_key=subkey,
                                        )
            dataset['state_trajectories'] = jnp.concatenate(
                    (dataset['state_trajectories'], jnp.array([trajectory])), axis=0
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
    #################################################
    # For RLC circuit
    # AC = jnp.array([[0.0], [0.0], [1.0]])
    # AR = jnp.array([[1.0], [-1.0], [0.0]])
    # AL = jnp.array([[0.0], [1.0], [-1.0]])
    # AV = jnp.array([[1.0], [0.0], [0.0]])
    # AI = jnp.array([[0.0], [0.0], [0.0]])

    # R = 1
    # L = 1
    # C = 1

    # x0 = jnp.array([0.0, 0.0])
    # y0 = jnp.array([0.0, 0.0, 0.0, 0.0])
    # z0 = jnp.concatenate((x0, y0))
    # T = jnp.linspace(0, 1.5, 1000)

    # def r_func(delta_V, params=None):
    #     return delta_V / R
    
    # def q_func(delta_V, params=None):
    #     return C * delta_V
    
    # def grad_H_func(phi, params=None):
    #     return phi / L
    
    # def u_func(t, params):
    #     return jnp.array([jnp.sin(30 * t)])
    
    #################################################

    #################################################
    # For DC microgrid
    AC = jnp.array([[0.0, 0.0], 
                    [0.0, 0.0], 
                    [1.0, 0.0], 
                    [0.0, 0.0],         
                    [0.0, 1.0], 
                    [0.0, 0.0], 
                    [0.0, 0.0]])
    AR = jnp.array([[-1.0, 0.0, 0.0], 
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0]])
    AL = jnp.array([[0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0]])
    AV = jnp.array([[1.0, 0.0], 
                    [0.0, 0.0], 
                    [0.0, 0.0], 
                    [0.0, 0.0],         
                    [0.0, 0.0], 
                    [0.0, 0.0], 
                    [0.0, 1.0]])
    AI = jnp.array([[0.0, 0.0], 
                    [0.0, 0.0], 
                    [-1.0, 0.0], 
                    [0.0, 0.0],         
                    [0.0, -1.0], 
                    [0.0, 0.0], 
                    [0.0, 0.0]])
    
    R = jnp.array([0.2, 0.05, 0.2])
    L = jnp.array([1.8e-6, 1.8e-9, 1.8e-6])
    C = jnp.array([2.2e-6, 2.2e-6])

    # x0 = jnp.array([0.0, 0.0])
    # y0 = jnp.array([0.0, 0.0, 0.0, 0.0])
    # z0 = jnp.concatenate((x0, y0))
    # T = jnp.linspace(0, 1.5, 1000)

    def r_func(delta_V, params=None):
        return delta_V / R
    
    def q_func(delta_V, params=None):
        return C * delta_V
    
    def grad_H_func(phi, params=None):
        return phi / L
    
    def u_func(t, params):
        return jnp.array([0.8, 1.1, 100, 100])
    
    #################################################

    
    # seed = 42 # for training
    seed = 41 # for testing
    env = RLC_PH_DAE(AC, AR, AL, AV, AI, grad_H_func, q_func, r_func, u_func, dt=0.01)

    curdir = os.path.abspath(os.path.curdir)
    save_dir = os.path.abspath(os.path.join(curdir, 'rlc_dae_data'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    t = time.time()
    print('starting simulation')
    dataset = env.gen_dataset(
        z0_init_lb=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        z0_init_ub=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]),
        trajectory_num_steps=200, # 1000 for training, 200 for testing.
        num_trajectories=50, # 500 for training, 50 for testing
        save_str=save_dir,
    )
    print(time.time() - t)