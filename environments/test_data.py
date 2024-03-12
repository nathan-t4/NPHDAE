import numpy as np

data_path = 'results/double_mass_spring_data/Double_Spring_Mass_2024-03-09-09-18-47.pkl'
data = np.load(data_path, allow_pickle=True)
traj1 = data['state_trajectories'][0]
dt = data['config']['dt']

v01 = (traj1[1,0]-traj1[0,0]) / dt
v02 = (traj1[1,2]-traj1[0,2]) / dt

p01 = traj1[0,1]
p02 = traj1[0,3]

print(v01, p01)
print(v02, p02)