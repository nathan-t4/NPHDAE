import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
def plot(ts, pred_data, exp_data, aux_data, prefix, plot_dir, prediction='acceleration', show=False):
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    if prediction == 'acceleration':
        m = jnp.round(aux_data[0], 3)
        k = jnp.round(aux_data[1], 3)
        b = jnp.round(aux_data[2], 3)

        q0 = jnp.round(exp_data[0,0], 3)
        a0 = jnp.round(exp_data[1,0], 3)
        """ Error Plot """
        # fig, (ax1, ax2) = plt.subplots(2,1)
        # fig.suptitle(f'{prefix}: Eval Error \n q0 = {q0}, a0 = {a0}')

        # ax1.set_title('Position')
        # ax1.plot(ts, exp_data[0] - pred_data[0], label=[f'Mass {i}' for i in range(2)])
        # ax1.set_xlabel('Time [$s$]')
        # ax1.set_ylabel('Position error [$m$]')
        # ax1.legend()

        # ax2.set_title('Acceleration')
        # ax2.plot(ts, exp_data[1] - pred_data[1], label=[f'Mass {i}' for i in range(2)])
        # ax2.set_xlabel('Time [$s$]')
        # ax2.set_ylabel(r'Acceleration error [$\mu m/s^2$]')
        # ax2.legend()

        # plt.tight_layout()
        # fig.savefig(os.path.join(plot_dir, f'{prefix}_error.png'))
        # plt.show() if show else plt.close()

        for i in range(2):
            fig, (ax1, ax2) = plt.subplots(2,1)
            fig.suptitle(f'{prefix}: Mass {i} \n' + rf'$m_{i} = {m[i]}$, $k_{i} = {k[i]}$, $b_{i} = {b[i]}$')
            ax1.set_title(f'Position')
            ax1.plot(ts, pred_data[0,:,i], label='predicted')
            ax1.plot(ts, exp_data[0,:,i], label='expected')
            ax1.set_xlabel('Time [$s$]')
            ax1.set_ylabel('Position [$m$]')
            ax1.legend()

            ax2.set_title(f'Acceleration')
            ax2.plot(ts, pred_data[1,:,i], label='predicted')
            ax2.plot(ts, exp_data[1,:,i], label='expected')
            ax2.set_xlabel('Time [$s$]')
            ax2.set_ylabel(r'Acceleration [$\mu m/s^2$]')
            ax2.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'{prefix}_mass{i}.png'))
            if show: plt.show()
            plt.close()
    
    elif prediction == 'position':
        fig, ax1 = plt.subplots(1)

        ax1.set_title('Position')
        ax1.plot(ts, exp_data - pred_data, label=[f'Mass {i}' for i in range(2)])
        ax1.set_xlabel('Time [$s$]')
        ax1.set_ylabel(r'Position error [$\mu m$]')
        ax1.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(plot_dir, f'{prefix}_error.png'))
        plt.show() if show else plt.close()

        for i in range(2):
            fig, ax1 = plt.subplots(1)
            fig.suptitle(f'{prefix}: Mass {i}')
            ax1.set_title(f'Position')
            ax1.plot(ts, pred_data[:,i], label='predicted')
            ax1.plot(ts, exp_data[:,i], label='expected')
            ax1.set_xlabel('Time [$s$]')
            ax1.set_ylabel(r'Position [$\mu m$]')
            ax1.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'{prefix}_mass{i}.png'))
            if show: plt.show()
            plt.close()
        
    plt.close()