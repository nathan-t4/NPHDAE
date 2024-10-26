import sys, os
sys.path.append('../')
import jax.numpy as jnp

from model_instances.dc_microgrid_composite_dae import dc_microgrid_dae
from model_instances.dc_microgrid_composite_ndae import dc_microgrid_ndae
from common import compute_g_vals_along_traj, compute_traj_err

import matplotlib.pyplot as plt

# \definecolor{submodel1Color}{HTML}{E05F15}
# \definecolor{submodel2Color}{HTML}{07742D}
# \definecolor{compositeModelColor}{HTML}{4F359B}
# \definecolor{trueModelColor}{HTML}{130303}

# \definecolor{lightSubmodel1Color}{HTML}{F19B6A}
# \definecolor{lightSubmodel2Color}{HTML}{0cd452}
# \definecolor{lightCompositeModelColor}{HTML}{a494c4}

def compare_dc_microgrid(exp_file_name, regularization_method, reg_param):
    phndae_color = (12/255, 212/255, 82/255)
    mlp_color = (224/255, 95/255, 21/255)

    # title = '(implicit trapezoid, 200 ts)'
    title = '(rk4)'

    x0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
    y0 = jnp.array([100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # y0 = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    z0 = jnp.concatenate((x0, y0))
    T = jnp.linspace(0, 0.008, 800)
    # T = jnp.linspace(0, 0.004, 400)
    # T = jnp.linspace(0, 1.5, 1000)

    print("solving dae")
    dae_sol, composite_dae = dc_microgrid_dae(z0, T, regularization_method, reg_param)
    print("solving ndae")
    ndae_sol, composite_ndae = dc_microgrid_ndae(z0, T, exp_file_name, regularization_method, reg_param)

    # Plot only a certain variable
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(T, dae_sol[:,2], linewidth=8, color='black')
    ax.plot(T, ndae_sol[:,2], linewidth=5, color=phndae_color, linestyle='--')
    ax.grid()

    plt.savefig(f'microgrid_q1_{regularization_method}{reg_param}.png', dpi=600)
    plt.clf()
    plt.close()

    fig, axes = plt.subplots(6, 3, figsize=(20, 48))

    axes[0,0].plot(T, dae_sol[:,0], linewidth=5, color='black')
    axes[0,0].plot(T, ndae_sol[:,0], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$q_{1}$')

    axes[0,1].plot(T, dae_sol[:,1], linewidth=5, color='black')
    axes[0,1].plot(T, ndae_sol[:,1], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$q_{2}$')

    axes[0,2].plot(T, dae_sol[:,2], linewidth=5, color='black')
    axes[0,2].plot(T, ndae_sol[:,2], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$\phi_{1}$')

    axes[1,0].plot(T, dae_sol[:,3], linewidth=5, color='black')
    axes[1,0].plot(T, ndae_sol[:,3], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$\phi_{2}$')

    axes[1,1].plot(T, dae_sol[:,4], linewidth=5, color='black')
    axes[1,1].plot(T, ndae_sol[:,4], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$\phi_{3}$')

    axes[1,2].plot(T, dae_sol[:,5], linewidth=5, color='black')
    axes[1,2].plot(T, ndae_sol[:,5], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$e_{1}$')

    axes[2,0].plot(T, dae_sol[:,6], linewidth=5, color='black')
    axes[2,0].plot(T, ndae_sol[:,6], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$e_{2}$')

    axes[2,1].plot(T, dae_sol[:,7], linewidth=5, color='black')
    axes[2,1].plot(T, ndae_sol[:,7], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$e_{3}$')

    axes[2,2].plot(T, dae_sol[:,8], linewidth=5, color='black')
    axes[2,2].plot(T, ndae_sol[:,8], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$e_{4}$')

    axes[3,0].plot(T, dae_sol[:,9], linewidth=5, color='black')
    axes[3,0].plot(T, ndae_sol[:,9], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$e_{5}$')

    axes[3,1].plot(T, dae_sol[:,10], linewidth=5, color='black')
    axes[3,1].plot(T, ndae_sol[:,10], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$e_{6}$')

    axes[3,2].plot(T, dae_sol[:,11], linewidth=5, color='black')
    axes[3,2].plot(T, ndae_sol[:,11], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$e_{7}$')

    axes[4,0].plot(T, dae_sol[:,12], linewidth=5, color='black')
    axes[4,0].plot(T, ndae_sol[:,12], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$e_{8}$')

    axes[4,1].plot(T, dae_sol[:,13], linewidth=5, color='black')
    axes[4,1].plot(T, ndae_sol[:,13], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$e_{9}$')

    axes[4,2].plot(T, dae_sol[:,14], linewidth=5, color='black')
    axes[4,2].plot(T, ndae_sol[:,14], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$j_{v_{1}}$')

    axes[5,0].plot(T, dae_sol[:,15], linewidth=5, color='black')
    axes[5,0].plot(T, ndae_sol[:,15], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$j_{v_{2}}$')

    axes[5,1].plot(T, dae_sol[:,16], linewidth=5, color='black')
    axes[5,1].plot(T, ndae_sol[:,16], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$\lambda_{1}$')

    axes[5,2].plot(T, dae_sol[:,17], linewidth=5, color='black')
    axes[5,2].plot(T, ndae_sol[:,17], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$\lambda_{2}$')
    # fig.suptitle(f'Trajectory rollouts {title}')
    fig.tight_layout()
    plt.savefig(f'microgrid_traj_{regularization_method}{reg_param}.png', dpi=600)
    plt.clf()
    plt.close()

    # Plot g values
    gnorm, _ = compute_g_vals_along_traj(composite_dae.solver.g, [None, None, None], ndae_sol, T, num_diff_vars=5)

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(gnorm, color=phndae_color, linewidth=5)
    ax.grid()
    ax.set_title(f'Constraint violation norm {title}')
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$||g(x)||_2^2$')
    fig.tight_layout()
    plt.savefig(f'microgrid_g_vals_{regularization_method}{reg_param}.png', dpi=600)
    plt.clf()
    plt.close()

    # Compute trajectory error
    err = compute_traj_err(dae_sol, ndae_sol)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(err)
    ax.set_title(f'Trajectory error norm {title}')
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$||\hat{x} - x||_2^2$')
    fig.tight_layout()
    plt.savefig(f'microgrid_traj_err_{regularization_method}{reg_param}.png', dpi=600)
    plt.clf()
    plt.close()

    return jnp.mean(gnorm), jnp.mean(err)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--reg_method', type=str, default='none')
    parser.add_argument('--reg_param', type=float, default=0.0)
    args = parser.parse_args()

    exp_file_name = args.path
    regularization_method = args.reg_method
    reg_param = args.reg_param
    
    mean_gnorm, mean_err = compare_dc_microgrid(exp_file_name, regularization_method, reg_param)

    print("{} with lambda = {}: Mean gnorm {:.2f}. Mean trajectory error {:.2f}".format(regularization_method, reg_param, mean_gnorm, mean_err))