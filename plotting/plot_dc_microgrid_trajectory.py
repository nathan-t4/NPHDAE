import matplotlib.pyplot as plt
import pickle
import sys, os
sys.path.append('../')

data_path = os.path.abspath(os.path.join(os.path.curdir, '..', 'environments', 'rlc_dae_data'))
data_file_name = 'train_RLC_DAE_2024-08-04-18-31-33.pkl'
data_file_name = 'RLC_DAE_2024-08-06-11-05-02.pkl'
data_file_name = 'DGU_DAE_2024-08-06-15-47-18.pkl'

with open(os.path.join(data_path, data_file_name), 'rb') as f:
    dataset = pickle.load(f)

for traj in range(dataset['state_trajectories'].shape[0]):
    if (dataset['state_trajectories'][traj, 0, 2::] == 0).all():
        print('Traj {} BAD'.format(traj))

traj_ind = 100
num_timesteps = 1000

fig = plt.figure()

ax1 = fig.add_subplot(631)
ax1.plot(dataset['timesteps'][traj_ind, 0:num_timesteps], dataset['state_trajectories'][traj_ind, 0:num_timesteps, 0])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('q')

ax1 = fig.add_subplot(632)
ax1.plot(dataset['timesteps'][traj_ind, 0:num_timesteps], dataset['state_trajectories'][traj_ind, 0:num_timesteps, 1])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('phi')

ax1 = fig.add_subplot(633)
ax1.plot(dataset['timesteps'][traj_ind, 0:num_timesteps], dataset['state_trajectories'][traj_ind, 0:num_timesteps, 2])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('e1')

ax1 = fig.add_subplot(634)
ax1.plot(dataset['timesteps'][traj_ind, 0:num_timesteps], dataset['state_trajectories'][traj_ind, 0:num_timesteps, 3])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('e2')

ax1 = fig.add_subplot(635)
ax1.plot(dataset['timesteps'][traj_ind, 0:num_timesteps], dataset['state_trajectories'][traj_ind, 0:num_timesteps, 4])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('e3')

ax1 = fig.add_subplot(636)
ax1.plot(dataset['timesteps'][traj_ind, 0:num_timesteps], dataset['state_trajectories'][traj_ind, 0:num_timesteps, 5])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

ax1 = fig.add_subplot(637)
ax1.plot(dataset['timesteps'][traj_ind, 0:num_timesteps], dataset['state_trajectories'][traj_ind, 0:num_timesteps, 6])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

ax1 = fig.add_subplot(638)
ax1.plot(dataset['timesteps'][traj_ind, 0:num_timesteps], dataset['state_trajectories'][traj_ind, 0:num_timesteps, 7])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

ax1 = fig.add_subplot(639)
ax1.plot(dataset['timesteps'][traj_ind, 0:num_timesteps], dataset['state_trajectories'][traj_ind, 0:num_timesteps, 8])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

ax1 = fig.add_subplot(6,3,10)
ax1.plot(dataset['timesteps'][traj_ind, 0:num_timesteps], dataset['state_trajectories'][traj_ind, 0:num_timesteps, 9])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

ax1 = fig.add_subplot(6,3,11)
ax1.plot(dataset['timesteps'][traj_ind, 0:num_timesteps], dataset['state_trajectories'][traj_ind, 0:num_timesteps, 10])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

ax1 = fig.add_subplot(6,3,12)
ax1.plot(dataset['timesteps'][traj_ind, 0:num_timesteps], dataset['state_trajectories'][traj_ind, 0:num_timesteps, 11])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

ax1 = fig.add_subplot(6,3,13)
ax1.plot(dataset['timesteps'][traj_ind, 0:num_timesteps], dataset['state_trajectories'][traj_ind, 0:num_timesteps, 12])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

ax1 = fig.add_subplot(6,3,14)
ax1.plot(dataset['timesteps'][traj_ind, 0:num_timesteps], dataset['state_trajectories'][traj_ind, 0:num_timesteps, 13])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

ax1 = fig.add_subplot(6,3,15)
ax1.plot(dataset['timesteps'][traj_ind, 0:num_timesteps], dataset['state_trajectories'][traj_ind, 0:num_timesteps, 14])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

ax1 = fig.add_subplot(6,3,16)
ax1.plot(dataset['timesteps'][traj_ind, 0:num_timesteps], dataset['state_trajectories'][traj_ind, 0:num_timesteps, 15])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

ax1 = fig.add_subplot(6,3,17)
ax1.plot(dataset['timesteps'][traj_ind, 0:num_timesteps], dataset['state_trajectories'][traj_ind, 0:num_timesteps, 16])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

ax1 = fig.add_subplot(6,3,18)
ax1.plot(dataset['timesteps'][traj_ind, 0:num_timesteps], dataset['state_trajectories'][traj_ind, 0:num_timesteps, 17])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('jv')

plt.savefig('dc_microgrid_trajectory.png')