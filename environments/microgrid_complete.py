import os
import numpy as np
import jax.numpy as jnp
import time
from environments.dgu_triangle_dae import DGU_TRIANGLE_PH_DAE
from environments.dgu_triangle_dae import generate_dataset as test_generate_dataset

def generate_dataset(num_dgu=3, ntraj=1, ntimesteps=150, dt=1e-2, z0=None, system_params=None, plot=True):
    """ 
        Simulate DC microgrid in a complete graph configuration with num_dgu vertices 
        The nodes are the DGUs and the edges are transmission lines
    """
    num_transmission_lines = int(0.5 * (num_dgu * (num_dgu - 1)))
    num_nodes = 3 * num_dgu + num_transmission_lines
    num_capacitors = num_dgu
    num_inductors = num_dgu + num_transmission_lines
    num_volt_sources = num_dgu
    num_cur_sources = num_dgu
    num_resistors = num_dgu + num_transmission_lines

    if system_params is None:
        R = 1
        L = 1
        C = 1
        control = [0.1] * num_cur_sources + [1.0] * num_volt_sources
    
    else:
        # TODO: the functions cannot take lists as inputs
        R = system_params['R']
        L = system_params['L']
        C = system_params['C']
        i_load = system_params['i_load'] # list
        v = system_params['v'] # list
        control = i_load + v

    AC = np.zeros((num_nodes,num_capacitors))
    for i in range(num_capacitors):
        AC[3 * i + 2, i] = 1

    AR = np.zeros((num_nodes,num_resistors))
    for i in range(num_dgu):
        idx = 3 * i
        AR[idx:idx+2,i] = [-1,1] # for DGU
    
    k = 0
    for i in range(num_dgu - 1):
        dgu_node_idx = 3 * i + 2
        for j in range(num_dgu - i - 1):
            tl_node_idx = 3*num_dgu + k
            resistor_idx = num_dgu + k
            AR[dgu_node_idx,resistor_idx] = 1
            AR[tl_node_idx,resistor_idx] = -1
            k = k + 1

    AL = np.zeros((num_nodes,num_inductors))
    for i in range(num_dgu):
        idx = 3 * i + 1
        AL[idx:idx+2,i] = [1,-1]

    k = 0
    for i in range(num_dgu - 1):
        for j in range(num_dgu - i - 1):
            start_idx = 3 * num_dgu + k
            end_idx = 3 * (i + j + 1) + 2
            inductor_idx = num_dgu + k
            AL[end_idx,inductor_idx] = 1
            AL[start_idx,inductor_idx] = -1
            k = k + 1
    

    AV = np.zeros((num_nodes,num_volt_sources))
    for i in range(num_dgu):
        idx = 3 * i
        AV[idx,i] = 1
    

    AI = np.zeros((num_nodes,num_cur_sources))
    for i in range(num_dgu):
        idx = 3 * i + 2
        AI[idx,i] = -1


    # AC_t = np.zeros((num_nodes,3))
    # AC_t[2,0] = 1
    # AC_t[5,1] = 1
    # AC_t[8,2] = 1

    # AR_t = np.zeros((num_nodes,6))
    # AR_t[0:2,0] = [-1,1]
    # AR_t[3:5,1] = [-1,1]
    # AR_t[6:8,2] = [-1,1]
    # AR_t[2,3] = 1; AR_t[9,3] = -1
    # AR_t[2,4] = 1; AR_t[10,4] = -1
    # AR_t[5,5] = 1; AR_t[11,5] = -1

    # AL_t = np.zeros((num_nodes,6))
    # AL_t[1:3,0] = [1,-1]
    # AL_t[4:6,1] = [1,-1]
    # AL_t[7:9,2] = [1,-1]
    # AL_t[5,3] = 1; AL_t[9,3] = -1
    # AL_t[8,4] = 1; AL_t[10,4] = -1
    # AL_t[8,5] = 1; AL_t[11,5] = -1

    # AV_t = np.zeros((num_nodes,3))
    # AV_t[0,0] = 1
    # AV_t[3,1] = 1
    # AV_t[6,2] = 1

    # AI_t = np.zeros((num_nodes,3))
    # AI_t[2,0] = -1
    # AI_t[5,1] = -1
    # AI_t[8,2] = -1

    # assert(AC == AC_t).all()
    # assert(AR == AR_t).all()
    # print("##########################")
    # print(AL)
    # print(AL_t)
    # print("##########################")
    # # assert(AL == AL_t).all()
    # assert(AV == AV_t).all()
    # assert(AI == AI_t).all()

    def r_func(delta_V, params=None):
        return delta_V / R
    
    def q_func(delta_V, params=None):
        return C * delta_V
    
    def grad_H_func(phi, params=None):
        return phi / L
    
    def u_func(t, params):
        return jnp.array(control)
    
    seed = 30 # for testing
    seed = 1 # for training
    env = DGU_TRIANGLE_PH_DAE(AC, AR, AL, AV, AI, grad_H_func, q_func, r_func, u_func, dt=dt, seed=seed)

    curdir = os.path.abspath(os.path.curdir)
    save_dir = os.path.abspath(os.path.join(curdir, 'dgu_triangle_data'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    t = time.time()
    print('starting simulation')
    if z0 is None:
        z0_init_lb = jnp.concatenate((-0.1 * jnp.ones(num_capacitors+num_inductors), jnp.zeros(num_nodes+num_volt_sources)))
        z0_init_ub = jnp.concatenate((0.1 * jnp.ones(num_capacitors+num_inductors), jnp.ones(num_nodes+num_volt_sources)))
    else:
        z0_init_lb = z0_init_ub = z0
    
    dataset = env.gen_dataset(
        z0_init_lb=z0_init_lb,
        z0_init_ub=z0_init_ub,
        trajectory_num_steps=ntimesteps, # 700 for training, 800 for testing.
        num_trajectories=ntraj, # 500 for training, 20 for testing
        save_str=save_dir,
    )

    z0 = dataset['state_trajectories'][0,0]
    print(z0.shape)
    # test_dataset = test_generate_dataset(ntraj, ntimesteps, dt, z0, False)
    
    # print('init condition error', z0 - test_dataset['state_trajectories'][0,0])

    import matplotlib.pyplot as plt
    if plot:
        traj = dataset['state_trajectories'][0,:,:]
        # test_traj = test_dataset['state_trajectories'][0,:,:]
        T = jnp.arange(traj.shape[0]) * env.dt
        fig, ax = plt.subplots(2, 2, figsize=(15,25))
        
        def get_labels():
            diff_states_idx = np.arange(0,num_capacitors+num_inductors)
            alg_states_idx = np.arange(num_capacitors+num_inductors, num_capacitors+num_inductors+num_nodes+num_volt_sources)
            q_labels = [f'q{i}' for i in range(num_capacitors)]
            phi_labels = [f'phi{i}' for i in range(num_inductors)]
            e_labels = [f'e{i}' for i in range(num_nodes)]
            jv_labels = [f'jv{i}' for i in range(num_volt_sources)]
            diff_labels = np.concatenate((q_labels, phi_labels))
            alg_labels = np.concatenate((e_labels, jv_labels))
            return diff_states_idx, alg_states_idx, diff_labels, alg_labels

        diff_states_idx, alg_states_idx, diff_labels, alg_labels = get_labels()

        ax[0,0].plot(T, traj[:,jnp.array(diff_states_idx[:num_capacitors])], label=diff_labels[:num_capacitors])
        ax[0,1].plot(T, traj[:,jnp.array(diff_states_idx[num_capacitors:])], label=diff_labels[num_capacitors:])
        ax[1,0].plot(T, traj[:,jnp.array(alg_states_idx[:num_nodes])], label=alg_labels[:num_nodes])
        ax[1,1].plot(T, traj[:,jnp.array(alg_states_idx[num_nodes:])], label=alg_labels[num_nodes:])
        ax[0,0].legend(loc='upper right')
        ax[0,1].legend(loc='upper right')
        ax[1,0].legend(loc='upper right')
        ax[1,1].legend(loc='upper right')
        plt.savefig(f'dgu_{num_dgu}_actual.png')
        plt.show()
        plt.close()

        # fig, ax = plt.subplots(2, 2, figsize=(15,25))
        # ax[0,0].plot(T, traj[:,jnp.array(diff_states_idx[:num_capacitors])] - test_traj[:,jnp.array(diff_states_idx[:num_capacitors])], label=diff_labels[:num_capacitors])
        # plt.tight_layout()
        # plt.show()
        # plt.close()
        

    print(time.time() - t)

    return dataset

if __name__ == '__main__':
    generate_dataset(num_dgu=5, ntimesteps=5000, dt=1e-3)