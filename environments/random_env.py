import os
import jax
import jax.numpy as jnp
import pickle
from datetime import datetime
from tqdm import tqdm
from models.ph_dae_random import RandomPHDAE

class PHDAEEnvironment():

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
            name='PHDAEEnv'
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
                    save_str=None,
                    save_name=None):
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
        for traj_ind in tqdm(range(1, num_trajectories), desc='Generating training trajectories'):
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
        
        if dataset['control_inputs'].shape == 0:
            dataset.pop('control_inputs')
        
        if save_str is not None:
            assert os.path.isdir(save_str)
            if save_name is None:
                save_name = os.path.join(
                    os.path.abspath(save_str),  
                    datetime.now().strftime(self.name + '_%Y-%m-%d-%H-%M-%S.pkl')
                )
            # jnp.save(save_path, dataset)
            with open(save_name, 'wb') as f:
                pickle.dump(dataset, f)

        return dataset