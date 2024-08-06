import matplotlib.pyplot as plt
import pickle
import sys, os
sys.path.append('../')

data_path = os.path.abspath(os.path.join(os.path.curdir, '..', 'environments', 'rlc_dae_data'))
data_file_name = 'train_RLC_DAE_2024-08-04-18-31-33.pkl'
data_file_name = 'RLC_DAE_2024-08-06-11-05-02.pkl'

with open(os.path.join(data_path, data_file_name), 'rb') as f:
    dataset = pickle.load(f)

for traj in range(dataset['state_trajectories'].shape[0]):
    if (dataset['state_trajectories'][traj, 0, 2::] == 0).all():
        print('Traj {} BAD'.format(traj))

traj_ind = 100
num_timesteps = 1000

fig = plt.figure()

ax1 = fig.add_subplot(321)
ax1.plot(dataset['timesteps'][traj_ind, 0:num_timesteps], dataset['state_trajectories'][traj_ind, 0:num_timesteps, 0])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('q')

ax1 = fig.add_subplot(322)
ax1.plot(dataset['timesteps'][traj_ind, 0:num_timesteps], dataset['state_trajectories'][traj_ind, 0:num_timesteps, 1])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('phi')

ax1 = fig.add_subplot(323)
ax1.plot(dataset['timesteps'][traj_ind, 0:num_timesteps], dataset['state_trajectories'][traj_ind, 0:num_timesteps, 2])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('e1')

ax1 = fig.add_subplot(324)
ax1.plot(dataset['timesteps'][traj_ind, 0:num_timesteps], dataset['state_trajectories'][traj_ind, 0:num_timesteps, 3])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('e2')

ax1 = fig.add_subplot(325)
ax1.plot(dataset['timesteps'][traj_ind, 0:num_timesteps], dataset['state_trajectories'][traj_ind, 0:num_timesteps, 4])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('e3')

ax1 = fig.add_subplot(326)
ax1.plot(dataset['timesteps'][traj_ind, 0:num_timesteps], dataset['state_trajectories'][traj_ind, 0:num_timesteps, 5])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

plt.show()