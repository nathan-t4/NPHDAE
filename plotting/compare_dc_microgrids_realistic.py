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

def compare_dc_microgrid(regularization_method, reg_param):
    phndae_color = (12/255, 212/255, 82/255)
    mlp_color = (224/255, 95/255, 21/255)

    x0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
    y0 = jnp.array([100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # y0 = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    z0 = jnp.concatenate((x0, y0))
    # T = jnp.linspace(0, 800*1e-8, 800)
    T = jnp.linspace(0, 0.008, 800)
    # T = jnp.linspace(0, 1.5, 1000)

    print("solving dae")
    dae_sol, composite_dae = dc_microgrid_dae(z0, T, regularization_method, reg_param)
    print("solving ndae")
    ndae_sol, composite_ndae = dc_microgrid_ndae(z0, T, regularization_method, reg_param)

    # Plot only a certain variable
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(T, dae_sol[:,2], linewidth=8, color='black')
    ax.plot(T, ndae_sol[:,2], linewidth=5, color=phndae_color, linestyle='--')
    ax.grid()

    plt.savefig(f'microgrid_q1_{regularization_method}{reg_param}.png', dpi=600)

    fig = plt.figure(figsize=(20, 48))

    ax1 = fig.add_subplot(631)
    ax1.plot(T, dae_sol[:,0], linewidth=5, color='black')
    ax1.plot(T, ndae_sol[:,0], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$q_{1}$')

    ax1 = fig.add_subplot(632)
    ax1.plot(T, dae_sol[:,1], linewidth=5, color='black')
    ax1.plot(T, ndae_sol[:,1], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$q_{2}$')

    ax1 = fig.add_subplot(633)
    ax1.plot(T, dae_sol[:,2], linewidth=5, color='black')
    ax1.plot(T, ndae_sol[:,2], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$\phi_{1}$')

    ax1 = fig.add_subplot(634)
    ax1.plot(T, dae_sol[:,3], linewidth=5, color='black')
    ax1.plot(T, ndae_sol[:,3], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$\phi_{2}$')

    ax1 = fig.add_subplot(635)
    ax1.plot(T, dae_sol[:,4], linewidth=5, color='black')
    ax1.plot(T, ndae_sol[:,4], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$\phi_{3}$')

    ax1 = fig.add_subplot(636)
    ax1.plot(T, dae_sol[:,5], linewidth=5, color='black')
    ax1.plot(T, ndae_sol[:,5], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$e_{1}$')

    ax1 = fig.add_subplot(637)
    ax1.plot(T, dae_sol[:,6], linewidth=5, color='black')
    ax1.plot(T, ndae_sol[:,6], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$e_{2}$')

    ax1 = fig.add_subplot(638)
    ax1.plot(T, dae_sol[:,7], linewidth=5, color='black')
    ax1.plot(T, ndae_sol[:,7], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$e_{3}$')

    ax1 = fig.add_subplot(639)
    ax1.plot(T, dae_sol[:,8], linewidth=5, color='black')
    ax1.plot(T, ndae_sol[:,8], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$e_{4}$')

    ax1 = fig.add_subplot(6,3,10)
    ax1.plot(T, dae_sol[:,9], linewidth=5, color='black')
    ax1.plot(T, ndae_sol[:,9], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$e_{5}$')

    ax1 = fig.add_subplot(6,3,11)
    ax1.plot(T, dae_sol[:,10], linewidth=5, color='black')
    ax1.plot(T, ndae_sol[:,10], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$e_{6}$')

    ax1 = fig.add_subplot(6,3,12)
    ax1.plot(T, dae_sol[:,11], linewidth=5, color='black')
    ax1.plot(T, ndae_sol[:,11], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$e_{7}$')

    ax1 = fig.add_subplot(6,3,13)
    ax1.plot(T, dae_sol[:,12], linewidth=5, color='black')
    ax1.plot(T, ndae_sol[:,12], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$e_{8}$')

    ax1 = fig.add_subplot(6,3,14)
    ax1.plot(T, dae_sol[:,13], linewidth=5, color='black')
    ax1.plot(T, ndae_sol[:,13], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$e_{9}$')

    ax1 = fig.add_subplot(6,3,15)
    ax1.plot(T, dae_sol[:,14], linewidth=5, color='black')
    ax1.plot(T, ndae_sol[:,14], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$j_{v_{1}}$')

    ax1 = fig.add_subplot(6,3,16)
    ax1.plot(T, dae_sol[:,15], linewidth=5, color='black')
    ax1.plot(T, ndae_sol[:,15], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$j_{v_{2}}$')

    ax1 = fig.add_subplot(6,3,17)
    ax1.plot(T, dae_sol[:,16], linewidth=5, color='black')
    ax1.plot(T, ndae_sol[:,16], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$\lambda_{1}$')

    ax1 = fig.add_subplot(6,3,18)
    ax1.plot(T, dae_sol[:,17], linewidth=5, color='black')
    ax1.plot(T, ndae_sol[:,17], linewidth=3, color=phndae_color, linestyle='--')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$\lambda_{2}$')

    plt.savefig(f'microgrid_traj_{regularization_method}{reg_param}.png', dpi=600)
    plt.clf()

    # Plot g values
    gnorm, _ = compute_g_vals_along_traj(composite_dae.solver.g, [None, None, None], ndae_sol, T, num_diff_vars=5)

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(gnorm, color=phndae_color, linewidth=5)
    ax.grid()
    plt.savefig(f'microgrid_g_vals_{regularization_method}{reg_param}.png', dpi=600)
    plt.clf()

    # Compute trajectory error
    err = compute_traj_err(dae_sol, ndae_sol)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(err)
    plt.savefig(f'microgrid_traj_err_{regularization_method}{reg_param}.png', dpi=600)
    plt.clf()

    return jnp.mean(gnorm), jnp.mean(err)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--reg_method', type=str, default='none')
    parser.add_argument('--reg_param', type=float, default=0.0)
    args = parser.parse_args()

    regularization_method = args.reg_method
    reg_param = args.reg_param

    mean_gnorm, mean_err = compare_dc_microgrid(regularization_method, reg_param)

    print("{} with lambda = {}: Mean gnorm {:.2f}. Mean trajectory error {:.2f}".format(regularization_method, reg_param, mean_gnorm, mean_err))