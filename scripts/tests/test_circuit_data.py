import numpy as np
import matplotlib.pyplot as plt
path = 'results/CoupledLC_data/val_20_1500.pkl'
data = np.load(path, allow_pickle=1)

trajs = data['state_trajectories']

mins = []
maxs = []
for i in range(trajs.shape[2]):
    maxs.append(trajs[:,:,i].max())
    mins.append(trajs[:,:,i].min())

print(maxs)
print(mins)

trajs = data['state_trajectories']
T = np.arange(trajs.shape[1])

plt.plot(T, trajs[8])
plt.show()