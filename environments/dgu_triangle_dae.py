import sys

sys.path.append('../')
from jax._src.random import PRNGKey as PRNGKey
import numpy as np
import jax.numpy as jnp
import jax

from models.ph_dae import PHDAE
import os
from datetime import datetime
import pickle
import time
import tqdm

import matplotlib.pyplot as plt

class DGU_TRIANGLE_PH_DAE():

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
        control_inputs = jax.vmap(self.dae.u_func, in_axes=(0,None))(T, None)
        # Generate the trajectory. Note that the dae solver automatically
        # finds the closest initial state for the algebraic variables
        return self.dae.solve(z0val, T, self.params), T, control_inputs

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
        trajectory, timesteps, control_inputs = self.gen_random_trajectory(
                                        z0_init_lb, 
                                        z0_init_ub, 
                                        trajectory_num_steps=\
                                            trajectory_num_steps,
                                        rng_key = subkey,
                                    )
        dataset['state_trajectories'] = jnp.array([trajectory])
        dataset['timesteps'] = jnp.array([timesteps])
        dataset['control_inputs'] = jnp.array([control_inputs])

        # training_dataset = jnp.array([jnp.stack((state, next_state), axis=0)])
        for traj_ind in tqdm.tqdm(range(1, num_trajectories), desc='Generating training trajectories'):
            self._rng_key, subkey = jax.random.split(self._rng_key)
            trajectory, timesteps, control_inputs = self.gen_random_trajectory(
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
            dataset['control_inputs'] = jnp.concatenate(
                    (dataset['control_inputs'], jnp.array([control_inputs])), axis=0
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
    num_nodes = 12
    AC = np.zeros((num_nodes,3))
    AC[2,0] = 1
    AC[5,1] = 1
    AC[8,2] = 1

    AR = np.zeros((num_nodes,6))
    AR[0:2,0] = [-1,1]
    AR[3:5,1] = [-1,1]
    AR[6:8,2] = [-1,1]
    AR[2,3] = 1; AR[9,3] = -1
    AR[2,4] = 1; AR[10,4] = -1
    AR[5,5] = 1; AR[11,5] = -1

    AL = np.zeros((num_nodes,6))
    AL[1:3,0] = [1,-1]
    AL[4:6,1] = [1,-1]
    AL[7:9,2] = [1,-1]
    AL[5,3] = 1; AL[9,3] = -1
    AL[8,4] = 1; AL[10,4] = -1
    AL[8,5] = 1; AL[11,5] = -1

    AV = np.zeros((num_nodes,3))
    AV[0,0] = 1
    AV[3,1] = 1
    AV[6,2] = 1

    AI = np.zeros((num_nodes,3))
    AI[2,0] = -1
    AI[5,1] = -1
    AI[8,2] = -1


    R = 1
    L = 1
    C = 1

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
        return jnp.array([0.1, 0.1, 0.1, 1.0, 1.0, 1.0])
    
    # seed = 42 # for testing
    seed = 41 # for training
    env = DGU_TRIANGLE_PH_DAE(AC, AR, AL, AV, AI, grad_H_func, q_func, r_func, u_func, dt=0.01)

    curdir = os.path.abspath(os.path.curdir)
    save_dir = os.path.abspath(os.path.join(curdir, 'dgu_triangle_data'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    t = time.time()
    print('starting simulation')
    dataset = env.gen_dataset(
        z0_init_lb=jnp.concatenate((1 * jnp.zeros(9), jnp.zeros(15))),
        z0_init_ub=jnp.concatenate((1 * jnp.zeros(9), jnp.zeros(15))),
        trajectory_num_steps=150, # 700 for training, 800 for testing.
        num_trajectories=1, # 500 for training, 20 for testing
        save_str=save_dir,
    )

    import matplotlib.pyplot as plt
    T = jnp.arange(150)
    traj = dataset['state_trajectories'][0,:,:]
    fig, ax = plt.subplots(2, figsize=(15,25))
    ax[0].plot(T, traj[:,jnp.array([0,1,2,3,4,5,6,7,8,21,22,23])], label=['q1', 'q2', 'q3', 'phi1', 'phi2', 'phi_3', 'phi_4', 'phi_5', 'phi_6', 'jv1', 'jv2', 'jv3'])
    ax[1].plot(T, traj[:,9:21], label=['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10', 'e11', 'e12'])
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    plt.savefig('dgu_triangle_actual.png')
    plt.show()

    print(time.time() - t)