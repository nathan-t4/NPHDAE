import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from GridCalEngine.api import *
from GridCalEngine.DataStructures.numerical_circuit import compile_numerical_circuit_at

####################################################################################################################
# Define the circuit
#
# A circuit contains all the grid information regardless of the islands formed or the amount of devices
####################################################################################################################

# create a circuit

grid = MultiCircuit(name='lynn 5 bus')

# let's create a master profile
T = 100
date0 = dt.datetime(2021, 1, 1)
time_array = pd.DatetimeIndex([date0 + dt.timedelta(hours=i) for i in range(T)])
x = np.linspace(-np.pi, np.pi, len(time_array))
y = np.abs(np.sin(x))
df_0 = pd.DataFrame(data=y, index=time_array)  # complex values

# set the grid master time profile
grid.time_profile = df_0.index

####################################################################################################################
# Define the buses
####################################################################################################################
bus1 = Bus(name='Bus1')
bus2 = Bus(name='Bus2')
bus3 = Bus(name='Bus3')
bus4 = Bus(name='Bus4')
bus5 = Bus(name='Bus5')

# add the bus objects to the circuit
grid.add_bus(bus1)
grid.add_bus(bus2)
grid.add_bus(bus3)
grid.add_bus(bus4)
grid.add_bus(bus5)

####################################################################################################################
# Add the loads
####################################################################################################################

# In GridCal, the loads, generators ect are stored within each bus object:

# we'll define the first load completely
l2 = Load(name='Load',
          G=0, B=0,  # admittance of the ZIP model in MVA at the nominal voltage (MVA: Mega-volt Ampere)
          Ir=0, Ii=0,  # Current of the ZIP model in MVA at the nominal voltage
          P=40, Q=20,  # Power of the ZIP model in MVA
          active=True,  # Is active?
          mttf=0.0,  # Mean time to failure
          mttr=0.0  # Mean time to recovery
          )
# grid.add_load(bus2, l2)

# Define the others with the default parameters
grid.add_load(bus3, Load(P=25, Q=15))
grid.add_load(bus4, Load(P=40, Q=20))
grid.add_load(bus5, Load(P=50, Q=20))


####################################################################################################################
# Add the generators
####################################################################################################################

g1 = Generator('slack generator')
grid.add_generator(bus1, g1)
grid.add_generator(bus2, Generator(P=1)) # active power P

####################################################################################################################
# Add the lines
####################################################################################################################

grid.add_branch(Branch(bus1, bus2, name='Line 1-2', r=0.05, x=0.11, b=0.02, rate=50))
grid.add_branch(Branch(bus1, bus3, name='Line 1-3', r=0.05, x=0.11, b=0.02, rate=50))
grid.add_branch(Branch(bus1, bus5, name='Line 1-5', r=0.03, x=0.08, b=0.02, rate=80))
grid.add_branch(Branch(bus2, bus3, name='Line 2-3', r=0.04, x=0.09, b=0.02, rate=3))
grid.add_branch(Branch(bus2, bus5, name='Line 2-5', r=0.04, x=0.09, b=0.02, rate=10))
grid.add_branch(Branch(bus3, bus4, name='Line 3-4', r=0.06, x=0.13, b=0.03, rate=30))
grid.add_branch(Branch(bus4, bus5, name='Line 4-5', r=0.04, x=0.09, b=0.02, rate=30))

####################################################################################################################
# Overwrite the default profiles with the custom ones
####################################################################################################################

for gen in grid.get_static_generators():
    gen.P_prof = gen.Q * df_0.values[:, 0]
    gen.Q_prof = gen.Q * df_0.values[:, 0]


Q = []; P = []
for gen in grid.get_generators():
    gen.P_prof = gen.P * df_0.values[:, 0]
    P.append(gen.P_prof.dense_array)
    Q.append(np.zeros(len(time_array)))

for load in grid.get_loads():
    load.P_prof = load.P * df_0.values[:, 0] # real power
    load.Q_prof = load.Q * df_0.values[:, 0] # reactive power
    P.append(load.P_prof.dense_array)
    Q.append(load.Q_prof.dense_array)

P = [np.zeros(len(time_array)) if ar is None else ar for ar in P]
Q = np.array(Q).T; P = np.array(P).T


####################################################################################################################
# Run a power flow simulation
####################################################################################################################

# We need to specify power flow options
pf_options = PowerFlowOptions(solver_type=SolverType.NR,  # Base method to use
                              verbose=False,  # Verbose option where available
                              tolerance=1e-6,  # power error in p.u.
                              max_iter=25,  # maximum iteration number
                              control_q=True  # if to control the reactive power
                              )

# Declare and execute the power flow simulation
pf = PowerFlowDriver(grid, pf_options)
pf.run()

writer = pd.ExcelWriter('Results.xlsx')
# now, let's compose a nice DataFrame with the voltage results
headers = ['Vm (p.u.)', 'Va (Deg)', 'Vre', 'Vim']
Vm = np.abs(pf.results.voltage)
Va = np.angle(pf.results.voltage, deg=True)
Vre = pf.results.voltage.real
Vim = pf.results.voltage.imag
data = np.c_[Vm, Va, Vre, Vim]
v_df = pd.DataFrame(data=data, columns=headers, index=grid.get_bus_names())
# print('\n', v_df)
v_df.to_excel(writer, sheet_name='V')

# Let's do the same for the branch results
headers = ['Loading (%)', 'Power from (MVA)']
loading = np.abs(pf.results.loading) * 100
power = np.abs(pf.results.Sf)
data = np.c_[loading, power]
br_df = pd.DataFrame(data=data, columns=headers, index=grid.get_branch_names())
br_df.to_excel(writer, sheet_name='Br')

# Finally the execution metrics
print('\nError:', pf.results.error)
print('Elapsed time (s):', pf.results.elapsed, '\n')

####################################################################################################################
# Run a time series power flow simulation
####################################################################################################################

ts = PowerFlowTimeSeriesDriver(grid=grid, options=pf_options)
ts.run()

# print()
# print('-' * 200)
# print('Time series')
# print('-' * 200)
# print('Voltage time series')
# df_voltage = pd.DataFrame(data=np.abs(ts.results.voltage), columns=grid.get_bus_names(), index=grid.time_profile)
# df_voltage.to_excel(writer, sheet_name='Vts')
# print(df_voltage)

# writer.close()

plt.plot(np.arange(len(time_array)), np.abs(ts.results.voltage), label=grid.get_bus_names())
plt.legend()
plt.show()

grid.plot_graph()
plt.show()

######
nc = compile_numerical_circuit_at(circuit=grid)
Y = nc.get_admittance_matrices()
Ybus = Y.Ybus

B = np.imag(Ybus)
print(B.shape)

Results = 'Results.csv'
# Create Headers
headers = ['Vm (p.u.)', 'Va (Deg)', 'Vre', 'Vim']
# Choose variables to display
Cf, Ct, C = grid.get_bus_branch_connectivity_matrix()
print('Cf = ', Cf.todense()) # senders
print('Ct = ', Ct.todense()) # receivers

senders = Cf.toarray() @ np.arange(grid.get_bus_number()).reshape(-1,1)
receivers = Ct.toarray() @ np.arange(grid.get_bus_number()).reshape(-1,1)
senders = senders.flatten()
receivers = receivers.flatten()

print('Senders = ', senders)
print('Receivers = ', receivers)

# Bus
Vm = np.abs(ts.results.voltage) # voltage magnitude
Va = np.angle(ts.results.voltage, deg=True) # voltage angle
Vre = ts.results.voltage.real
Vim = ts.results.voltage.imag
# Branch
loading = np.abs(pf.results.loading)
power = np.abs(pf.results.Sf)

# Vm = np.expand_dims(Vm, 1)
# Va = np.expand_dims(Va, 1)
# Vre = np.expand_dims(Vre, 1)
# Vim = np.expand_dims(Vim, 1)

# data = np.concatenate([Vm, Va, Vre, Vim], axis=1)

# # Create Data Frame
# for i in range(len(headers)):
#     filename = headers[i] + '.csv'   
#     v_df = pd.DataFrame(data=data[:,i,:], columns=grid.get_bus_names(), index=grid.time_profile)
#     v_df.to_csv(filename)


H = [v.T @ B @ v + q.T @ np.log(V) + p.T @ theta for (q, p, v, V, theta) in zip(Q, P, Vim, Vm, Va)]
plt.figure()
plt.plot(np.arange(len(time_array)), H)
plt.show()

G = 1 # num of generators buses
L = 3 # num of load buses
S = 1 # num of slack bus

T1 = np.block([[-np.ones((G,1)), np.eye(G)],
               [-np.ones((L,1)), np.zeros((L,G))]]) # (G+L, 1+G)
T2 = np.block([[np.zeros((G,L))],
               [np.eye(L)]]) # (G+L, L)

M = np.eye(1+G) # (1+G, 1+G)
inv_M = np.linalg.inv(M)

D_g = np.diag(np.zeros(S+G)) # generator damping constants
D_d = np.diag(np.zeros(G+L)) # real load damping constants
D_e = np.diag(np.zeros(L))

# T1 @ inv_M: (G+L, 1+G), inv_M @ T1.T: (1+G, G+L)
# J (S+G+G+L+L, S+G+G+L+L)
J = np.block([[np.zeros((S+G,S+G)), -inv_M @ T1.T, np.zeros((S+G,L))],
              [T1 @ inv_M, np.zeros((G+L,G+L)), np.zeros((G+L,L))],
              [np.zeros((L,S+G)), np.zeros((L,G+L)), np.zeros((L,L))]])

# R = np.block([[inv_M @ D_g @ inv_M, np.zeros((S+G, G+L)), np.zeros((S+G,L))],
#               [np.zeros((G+L,S+G)), T2.T @ np.linalg.inv(D_d) @ T2.T, np.zeros(G+L,L)],
#               [np.zeros((L,S+G)), np.zeros((L,G+L)), np.linalg.inv(D_e) @ np.eye(L)]])
R = np.zeros((S+G+G+L+L, S+G+G+L+L))
print('J:', J)
print('R:', R)

data = {}

data['config'] = {
    'senders': senders,
    'receivers': receivers,
    'J': J,
    'R': R,
    'Y': Ybus,
    'B': B,
    'Q': Q,
    'P': P,
}

data['state_trajectories'] = {
    'Vm': Vm,
    'Va': Va,
    'Vre': Vre,
    'Vim': Vim,
    'H': H,
}

import pickle
import os
curdir = os.curdir
with open(os.path.join(curdir, 'results/grid_data/data.pkl'), 'wb') as f:
    pickle.dump(data, f)