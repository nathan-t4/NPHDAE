import sys, os
sys.path.append('../')
import numpy as np
import jax.numpy as jnp
from model_instances.microgrid_complete_dae import dc_microgrid_dae
from model_instances.microgrid_complete_ndae import dc_microgrid_ndae
from common import compute_g_vals_along_traj, compute_traj_err

import matplotlib.pyplot as plt

# \definecolor{submodel1Color}{HTML}{E05F15}
# \definecolor{submodel2Color}{HTML}{07742D}
# \definecolor{compositeModelColor}{HTML}{4F359B}
# \definecolor{trueModelColor}{HTML}{130303}

# \definecolor{lightSubmodel1Color}{HTML}{F19B6A}
# \definecolor{lightSubmodel2Color}{HTML}{0cd452}
# \definecolor{lightCompositeModelColor}{HTML}{a494c4}

def compare_dc_microgrid(exp_file_name, num_dgu):
    phndae_color = (12/255, 212/255, 82/255)
    mlp_color = (224/255, 95/255, 21/255)

    title = '(rk4)'

    num_tl = int(0.5 * (num_dgu * (num_dgu - 1)))
    num_capacitors = num_dgu
    num_inductors = num_dgu + num_tl
    num_nodes = 3 * num_dgu + 3 * num_tl # for ndae
    num_volt_sources = num_dgu
    num_couplings = 2 * num_tl

    seed = 42
    import jax.random as random
    key = random.key(seed)
    x0 = jnp.zeros(num_capacitors+num_inductors)
    x0 = 0.5 * (random.uniform(key, num_capacitors+num_inductors) * 2 - 1)
    y0 = jnp.zeros(num_nodes+num_volt_sources+num_couplings)
    z0 = jnp.concatenate((x0, y0))
    T = jnp.linspace(0, 0.008, 800)

    print(f"Number of states: {len(z0)}")
    print(f"Number of algebraic states: {len(y0)}")
    

    print("solving ndae")
    ndae_sol, composite_ndae, system_params = dc_microgrid_ndae(exp_file_name, num_dgu, z0, T, plot=False)

    print("solving dae")
    dae_sol, composite_dae = dc_microgrid_dae(num_dgu, z0, T, system_params, plot=False)

    # Plot only a certain variable
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(T, dae_sol[:,0], linewidth=8, color='black')
    ax.plot(T, ndae_sol[:,0], linewidth=5, color=phndae_color, linestyle='--')
    ax.grid()
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$q_1$')
    plt.savefig(f'microgrid_{num_dgu}_q1.png')
    plt.clf()
    plt.close()

    def get_labels(num_nodes):
        diff_states_idx = np.arange(0,num_capacitors+num_inductors)
        alg_states_idx = np.arange(num_capacitors+num_inductors, num_capacitors+num_inductors+num_nodes+num_volt_sources)
        q_labels = [f'q{i}' for i in range(num_capacitors)]
        phi_labels = [f'phi{i}' for i in range(num_inductors)]
        e_labels = [f'e{i}' for i in range(num_nodes)]
        jv_labels = [f'jv{i}' for i in range(num_volt_sources)]
        diff_labels = np.concatenate((q_labels, phi_labels))
        alg_labels = np.concatenate((e_labels, jv_labels))
        return diff_states_idx, alg_states_idx, diff_labels, alg_labels

    diff_states_idx, alg_states_idx, diff_labels, alg_labels = get_labels(num_nodes=num_nodes)
    fig, ax = plt.subplots(2, 2)
    ax[0,0].plot(T, 
                dae_sol[:,jnp.array(diff_states_idx[:num_capacitors])], 
                label=diff_labels[:num_capacitors],
                color='black')
    ax[0,0].plot(T, 
                ndae_sol[:,jnp.array(diff_states_idx[:num_capacitors])], 
                label=diff_labels[:num_capacitors],
                color=phndae_color,
                linestyle='--')
    ax[0,0].set_xlabel(r'$t$')
    ax[0,0].set_ylabel(r'$q$')

    ax[0,1].plot(T, 
                dae_sol[:,jnp.array(diff_states_idx[num_capacitors:])], 
                label=diff_labels[num_capacitors:],
                color='black')
    ax[0,1].plot(T,
                ndae_sol[:,jnp.array(diff_states_idx[num_capacitors:])],
                label=diff_labels[num_capacitors:],
                color=phndae_color,
                linestyle='--')
    ax[0,1].set_xlabel(r'$t$')
    ax[0,1].set_ylabel(r'$\phi$')

    ax[1,0].plot(T, 
                dae_sol[:,jnp.array(alg_states_idx[:num_nodes])], 
                label=alg_labels[:num_nodes],
                color='black')
    ax[1,0].plot(T,
                ndae_sol[:,jnp.array(alg_states_idx[:num_nodes])],
                label=alg_labels[:num_nodes],
                color=phndae_color,
                linestyle='--')
    ax[1,0].set_xlabel(r'$t$')
    ax[1,0].set_ylabel(r'$e$')

    ax[1,1].plot(T, 
                dae_sol[:,jnp.array(alg_states_idx[num_nodes:])], 
                label=alg_labels[num_nodes:],
                color='black')
    ax[1,1].plot(T,
                ndae_sol[:,jnp.array(alg_states_idx[num_nodes:])],
                label=alg_labels[num_nodes:],
                color=phndae_color,
                linestyle='--')
    ax[1,1].set_xlabel(r'$t$')
    ax[1,1].set_ylabel(r'$j_V$')

    # ax[0,0].legend(loc='upper right')
    # ax[0,1].legend(loc='upper right')
    # ax[1,0].legend(loc='upper right')
    # ax[1,1].legend(loc='upper right')
    # import matplotlib.patches as mpatches
    # dae_patch = mpatches.Patch(color='black', label='PHDAE')
    # ndae_patch = mpatches.Patch(color=phndae_color, label='PHNDAE')
    # ax[0,1].legend(handles=[dae_patch, ndae_patch], loc='upper right', bbox_to_anchor=(3,-1))

    plt.tight_layout()
    plt.savefig(f'microgrid_{num_dgu}_ndae_traj.png', dpi=600)
    plt.show()
    plt.clf()
    plt.close()

    # Compute trajectory error
    fig = plt.figure(figsize=(10,4))
    err = compute_traj_err(dae_sol, ndae_sol)
    err = err / (ndae_sol.shape[1]) # divide by dimension of state
    ax = fig.add_subplot(111)
    ax.plot(err)
    ax.set_title(f'Mean trajectory error norm ({num_dgu} DGUs)')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$||\hat{x} - x||_2^2$')
    fig.tight_layout()
    plt.savefig(f'microgrid_{num_dgu}_traj_err.png', dpi=600)
    plt.clf()
    plt.close()

    # Plot g values
    gnorm, gval = compute_g_vals_along_traj(composite_dae.solver.g, [None] * (num_dgu+num_tl), ndae_sol, T, num_diff_vars=num_capacitors+num_inductors)

    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111)
    ax.plot(T, gnorm, color=phndae_color, linewidth=5)
    ax.set_title(f'Mean constraint violation norm ({num_dgu} DGUs)')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$||g(\hat{x}) - g(x)||_2^2$')
    plt.savefig(f'microgrid_{num_dgu}_gnorm.png', dpi=600)
    plt.clf()
    plt.close()

    return jnp.mean(gnorm), jnp.mean(err)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--n', type=int, default=3)
    args = parser.parse_args()

    exp_file_name = args.path
    num_dgu = args.n
    
    mean_gnorm, mean_err = compare_dc_microgrid(exp_file_name, num_dgu)

    print("Mean gnorm {:.2f}. Mean trajectory error {:.2f}".format(mean_gnorm, mean_err))