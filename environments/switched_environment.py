from environments.environment import *
from typing import Dict

class SwitchedEnvironment(Environment):
    def __init__(self, dt=0.01, random_seed=42, name='switched_environment', integrator_name='rk4'):
        super().__init__(dt, random_seed, name, integrator_name)
        self.switchable_params = []
        self._define_switchable_params()

    
    def _define_step_function(self):
        def step(state: jnp.ndarray, t: jnp.float32, jax_key: jax.random.PRNGKey):
            key, control_key, dynamics_key = jax.random.split(jax_key, 3)
            control_input = self.control_policy(state, t, control_key)
            f = lambda x, t : self.dynamics_function(x, t, control_input, dynamics_key)

            next_state = self.integrator(f, state, t, self._dt)

            return next_state, control_input
        
        self.step = jax.jit(step)

        
    def _modify_config(self, params: Dict[str, jnp.float32]):
        # Update config
        for name,value in params.items():
            if name in self.switchable_params:
                previous_value = self.config[name]
                assert type(previous_value) == type(value), "TypeError: must update config with value of same type"
                self.config[name] = value
        # Update system parameters
        self._modify_params()
    
    @abstractmethod
    def _modify_params(self):
        """ Update system parameters based on config """
        raise NotImplementedError
    
    @abstractmethod
    def _define_switchable_params(self):
        """ Define list of switchable parameters """
        raise NotImplementedError

    def gen_trajectory(self, 
                        init_state : jnp.array,
                        trajectory_num_steps : int = 50,
                        switched_params: Dict[int, Dict[str, jnp.float32]] = None,
                        jax_key : jax.random.PRNGKey = None,
                        ) -> tuple:
        """
        Generate an individual system trajectory from a specific initial state.

        Parameters
        ----------
        init_state :
            Jax numpy array representing the initial system state.
        trajectory_num_steps : 
            Number of timesteps to include in the trajectory.

        Returns
        -------
        trajectory :
            Tuple of Jax numpy arrays. The arrays contain the same data, but 
            have a time offset of 1 step.
        """
        tIndexes = jnp.linspace(0, 
                                (trajectory_num_steps + 1) * self._dt, 
                                num=trajectory_num_steps + 1, 
                                endpoint=False, 
                                dtype=jnp.float32)
        # xnextVal = self.solve_analytical(init_state, tIndexes)

        # dyn_function = lambda state, t : self.dynamics_function(state, t, jnp.array([0.0]), None)
        # xnextVal = odeint(dyn_function, init_state, t=tIndexes, rtol=1e-10, atol=1e-10)
        # control_inputs = jnp.zeros((trajectory_num_steps, 1))

        # return xnextVal, tIndexes, control_inputs
        
        xnextVal = [init_state]
        control_inputs = []
        for steps, t in enumerate(tIndexes):
            jax_key, subkey = jax.random.split(jax_key)
            if steps in list(switched_params.keys()):
                self._modify_config(switched_params[steps])

            next_state, control = self.step(xnextVal[-1], t, subkey)
            control_inputs.append(control)
            xnextVal.append(next_state)

        # Append the last control input again to make the trajectory length the same.
        control_inputs.append(control)

        xnextVal = jnp.array(xnextVal[:-1])
        control_inputs = jnp.array(control_inputs[:-1])
        return xnextVal, tIndexes, control_inputs       

    def gen_random_trajectory(self,
                                rng_key : jax.random.PRNGKey, 
                                x0_init_lb : jnp.array, 
                                x0_init_ub : jnp.array,
                                num_switches: int = 0,
                                trajectory_num_steps : int = 50,
                                ) -> tuple:
        """
        Generate a system trajectory from a random initial state.

        Parameters
        ----------
        rng_key :
            Jax PRNGkey
        x0_init_lb :
            Jax array representing lower bounds on the randomly selected 
            initial state.
        x0_init_ub : 
            Jax array representing upper bounds on the randomly selected 
            initial state.
        trajectory_num_steps : 
            Number of timesteps to include in the trajectory.

        Returns
        -------
        trajectory :
            Tuple of Jax numpy arrays. The arrays contain the same data, but 
            have a time offset of 1 step.
        """
        switched_params = {}
        for i in range(num_switches):
            # Switch one random parameter at random time
            key, subkey = jax.random.split(rng_key)
            rnd_timestep = np.random.randint(low=0, high=trajectory_num_steps)
            rnd_param_idx = np.random.randint(low=0, high=len(self.switchable_params))
            param_to_switch = self.switchable_params[rnd_param_idx]
            original_param_val = self.config[param_to_switch]
            switched_params[rnd_timestep] = {param_to_switch: 5.0 * np.random.random() * original_param_val}
        
        shape = x0_init_lb.shape
        key, subkey = jax.random.split(rng_key)
        x0val = jax.random.uniform(subkey, 
                                    shape=shape, 
                                    minval=x0_init_lb, 
                                    maxval=x0_init_ub)

        key, subkey = jax.random.split(key)
        return self.gen_trajectory(x0val, 
                                    trajectory_num_steps=trajectory_num_steps,
                                    switched_params=switched_params,
                                    jax_key = subkey)


    def gen_dataset(self,
                    x0_init_lb : jnp.array,
                    x0_init_ub : jnp.array,
                    num_switches_per_traj: int = 0,
                    trajectory_num_steps : int = 500,
                    num_trajectories : int = 200,
                    save_pixel_observations=False,
                    im_shape : tuple = (28,28),
                    grayscale : bool = True,
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
        dataset['pixel_trajectories'] = []

        self._rng_key, subkey = jax.random.split(self._rng_key)
        # TODO: switches occur at different random timesteps for each random traj (change to fixed switching)?
        trajectory, timesteps, control_inputs = self.gen_random_trajectory(subkey, 
                                                    x0_init_lb, 
                                                    x0_init_ub, 
                                                    num_switches=num_switches_per_traj,
                                                    trajectory_num_steps=\
                                                        trajectory_num_steps)
        dataset['state_trajectories'] = jnp.array([trajectory])
        dataset['timesteps'] = jnp.array([timesteps])
        dataset['control_inputs'] = jnp.array([control_inputs])

        if save_pixel_observations:
            pixel_trajectory = self.get_pixel_trajectory(trajectory, 
                                                        im_shape=im_shape, 
                                                        grayscale=grayscale)
            dataset['pixel_trajectories'].append(pixel_trajectory)

        # training_dataset = jnp.array([jnp.stack((state, next_state), axis=0)])
        for traj_ind in tqdm(range(1, num_trajectories), desc='Generating data'):
            self._rng_key, subkey = jax.random.split(self._rng_key)
            # TODO: switches occur at different random timesteps for each random traj (change to fixed switching)?
            trajectory, timesteps, control_inputs = self.gen_random_trajectory(subkey, 
                                                        x0_init_lb, 
                                                        x0_init_ub, 
                                                        num_switches=num_switches_per_traj,
                                                        trajectory_num_steps=\
                                                            trajectory_num_steps)
            dataset['state_trajectories'] = jnp.concatenate(
                    (dataset['state_trajectories'], jnp.array([trajectory])), axis=0
                )
            dataset['timesteps'] = jnp.concatenate(
                    (dataset['timesteps'], jnp.array([timesteps])), axis=0
                )
            dataset['control_inputs'] = jnp.concatenate(
                    (dataset['control_inputs'], 
                    jnp.array([control_inputs])), 
                    axis=0
                )

            if save_pixel_observations:
                pixel_trajectory = self.get_pixel_trajectory(trajectory, 
                                                            im_shape=im_shape, 
                                                            grayscale=grayscale)
                dataset['pixel_trajectories'].append(pixel_trajectory)
                    
        if save_str is not None:
            if not os.path.isdir(save_str):
                os.makedirs(save_str)
            save_path = os.path.join(os.path.abspath(save_str),  
                            datetime.now().strftime(f'{self.name}_{num_switches_per_traj}_switches_%Y-%m-%d-%H-%M-%S.pkl'))
            # jnp.save(save_path, dataset)
            with open(save_path, 'wb') as f:
                pickle.dump(dataset, f)

        return dataset
