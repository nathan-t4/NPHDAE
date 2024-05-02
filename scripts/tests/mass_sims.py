import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.integrate import solve_ivp, _ivp
ms = [1, 1, 1, 1]
ks = [1, 1, 1, 1]
ys = [1, 1, 1, 1]
def simulate_quadruple_mass_spring():
    """
        wall ---(ks[0], ys[0])--- 1 ---(ks[1], ys[1])--- 2 ---(ks[2], ys[2])--- 3 ---(ks[3], ys[3])--- 4
    """
    A = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                  [-(ks[0] + ks[1]) / ms[0], 0, ks[1] / ms[0], 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [ks[1] / ms[1], 0, -(ks[1] + ks[2]) / ms[1], 0, ks[2] / ms[1], 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, ks[2] / ms[2], 0, -(ks[2] + ks[3]) / ms[2], 0, ks[3] / ms[2], 0],
                  [0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, ks[3] / ms[3], 0, -ks[3] / ms[3], 0]])
    b = np.array([[0],
                  [(ks[0] * ys[0] - ks[1] * ys[1]) / ms[0]],
                  [0],
                  [(ks[1] * ys[1] - ks[2] * ys[2]) / ms[1]],
                  [0],
                  [(ks[2] * ys[2] - ks[3] * ys[3]) / ms[2]],
                  [0],
                  [(ks[3] * ys[3]) / ms[3]]])
    return A, b

def simulate_triple_mass_spring():
    # To compare triple mass spring with merging
    # TODO: check A, b
    A = np.array([[0, 1, 0, 0, 0, 0],
                  [-(ks[0] + ks[1]) / ms[0], 0, ks[1] / ms[0], 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [ks[1] / ms[1], 0, -(ks[1] + ks[2]) / ms[1], 0, ks[2] / ms[1], 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, ks[2] / ms[2], 0, -ks[2] / ms[2], 0]])
    b = np.array([[0],
                  [(ks[0] * ys[0] - ks[1] * ys[1]) / ms[0]],
                  [0],
                  [(ks[1] * ys[1] - ks[2] * ys[2]) / ms[1]],
                  [0],
                  [(ks[2] * ys[2]) / ms[2]]])
    return A, b

def simulate_double_mass_spring():
    """
        wall ---(ks[0], ys[0])--- 1 ---(ks[1], ys[1])--- 2
    """
    A = np.array([[0, 1, 0, 0],
                  [-(ks[0] + ks[1]) / ms[0], 0, ks[1] / ms[0], 0],
                  [0, 0, 0, 1],
                  [ks[1] / ms[1], 0, -ks[1] / ms[1], 0]])
    b = np.array([[0],
                  [ks[0] * ys[0] - ks[1] * ys[1]],
                  [0],
                  [ks[1] * ys[1]]])
    return A, b

def simulate_double_free_spring():
    """
        3 ---(ks[3], ys[3])--- 4
    """
    A = np.array([[0, 1, 0, 0],
                  [-ks[3] / ms[2], 0, ks[3] / ms[2], 0],
                  [0, 0, 0, 1],
                  [ks[3] / ms[3], 0, -ks[3] / ms[3], 0]])
    b = np.array([[0],
                  [-ks[3] * ys[3]],
                  [0],
                  [ks[3] * ys[3]]])
    return A, b

# Function implementing Runge-Kutta 4th order method
def runge_kutta_4th_order(f, t0, tf, x0, h):
    """
    f: function f(t, x) that returns dx/dt
    t0: initial time
    tf: final time
    x0: initial state
    h: step size
    """
    n = len(x0)
    t = np.arange(t0, tf, h)
    x = np.zeros((len(t), n))
    x[0] = x0
    for i in range(1, len(t)):
        k1 = h * f(t[i-1], x[i-1])
        k2 = h * f(t[i-1] + h/2, x[i-1] + k1/2)
        k3 = h * f(t[i-1] + h/2, x[i-1] + k2/2)
        k4 = h * f(t[i-1] + h, x[i-1] + k3)
        x[i] = x[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
    return t, x

# Function implementing euler integration
def euler_integrate(f, t0, tf, x0, h):
    """
    f: function f(t, x) that returns dx/dt
    t0: initial time
    tf: final time
    x0: initial state
    h: step size
    """
    n = len(x0)
    t = np.arange(t0, tf, h)
    x = np.zeros((len(t), n))
    x[0] = x0
    for i in range(1, len(t)):
        x[i] = x[i-1] + h * f(t[i-1], x[i-1])
    return t, x

def simulate(x_dot, T, x0, dt, args=None, show_plot=False):
    """ return states for t=0:T with initial condition x0 """
    # integrator = partial(_ivp.RK45, rtol=1e-6, atol=1e-16)
    sol = solve_ivp(x_dot, (0,T), x0, method='RK45', t_eval=np.arange(0,T,dt))
    t_out, x_out = euler_integrate(x_dot, 0, T, x0, dt)
    num_states = len(x0)
    N = num_states // 2
    labels = [''] * num_states
    labels[::2] = [f'q{i+1}' for i in range(N)]
    labels[1::2] = [f'q{i+1}dot' for i in range(N)]
    if show_plot:
        plt.plot(sol.t, sol.y.T, label=labels)
        plt.legend()
        plt.show()
    # return sol.t, sol.y.T, labels
    return t_out, x_out, labels

if __name__ == '__main__':
    dt = 0.1

    def test_adding_edges():
        T = 4
        x0 = np.array([1, 0, 2.2, 0, 2.8, 0, 4.3, 0])
        Ac, bc = simulate_quadruple_mass_spring()
        xc_dot = lambda t, x : Ac @ x + bc.flatten()
        tsc, ysc, labels = simulate(xc_dot, T, x0, dt)
        q2_traj = ysc[:,2]
        q3_traj = ysc[:,4]
        F23 = ks[2] * (q3_traj - q2_traj - ys[2])
        x10 = x0[:4]
        A1, b1 = simulate_double_mass_spring()
        x1_dot = lambda t, x : A1 @ x + b1.flatten()
        def x1_dot_with_force(t, x):
            if int(t) == T:
                F = 0
            else:
                F = F23[int(t / dt)]
            return A1 @ x + b1.flatten() + np.array([0, 0, 0, F / ms[2]])
        ts1, ys1, _ = simulate(x1_dot, T, x10, dt)
        ts1f, ys1f, _ = simulate(x1_dot_with_force, T, x10, dt)
        x20 = x0[4:]
        A2, b2 = simulate_double_free_spring()
        x2_dot = lambda t, x : A2 @ x + b2.flatten()
        def x2_dot_with_force(t, x):
            if int(t) == T:
                F = 0
            else:
                F = -F23[int(t / dt)]
            return A2 @ x + b2.flatten() + np.array([0, F / ms[3], 0, 0])
        ts2, ys2, _ = simulate(x2_dot, T, x20, dt)
        ts2f, ys2f, _ = simulate(x2_dot_with_force, T, x20, dt)
        axes = plt.figure(layout="constrained", figsize=(10,20)).subplot_mosaic(
            """
            AA
            BC
            DE
            FF
            """
        )
        axes['A'].set_title('[1] Composite system: quadruple mass-spring')
        axes['A'].plot(tsc, ysc, label=labels)
        axes['A'].legend(loc="upper right")
        axes['B'].set_title('[2] Subsystem 1: double mass-spring')
        axes['B'].plot(ts1, ys1, label=labels[:4])
        axes['B'].legend(loc="upper right")
        axes['C'].set_title('[3] Subsystem 1: double mass-spring + F12')
        axes['C'].plot(ts1f, ys1f, label=labels[:4])
        axes['C'].legend(loc="upper right")
        axes['D'].set_title('[4] Subsystem 2: double free-spring')
        axes['D'].plot(ts2, ys2, label=labels[4:])
        axes['D'].legend(loc="upper right")
        axes['E'].set_title('[5] Subsystem 2: double free-spring - F12')
        axes['E'].plot(ts2f, ys2f, label=labels[4:])
        axes['E'].legend(loc="upper right")
        errors = np.concatenate((ysc[:,:4] - ys1f, ysc[:,4:] - ys2f), axis=1)
        axes['F'].set_title('[6] Errors: composite_state - subsystem_state')
        axes['F'].plot(tsc, errors, label=labels)
        axes['F'].legend(loc="upper right")
        plt.savefig("test_add_edges.png")
        # Make a plot of F23
        plt.figure()
        plt.plot(tsc, F23)
        plt.xlabel('Time')
        plt.ylabel('F23')
        plt.title('F23 vs Time')
        plt.savefig('F23.png')

    def test_merging_nodes():
        T = 4
        x0 = [1, 0, 2.2, 0, 2.8, 0]

        Ac, bc = simulate_triple_mass_spring()
        xc_dot = lambda t, x : Ac @ x + bc.flatten()
        tsc, ysc, labels = simulate(xc_dot, T, x0, dt)

        x10 = x0[:4]
        A1, b1 = simulate_double_mass_spring()
        x1_dot = lambda t, x : A1 @ x + b1.flatten()
        ts1, ys1, _ = simulate(x1_dot, T, x10, dt)

        x20 = x0[2:]
        A2, b2 = simulate_double_free_spring() # make sure indices are correct
        x2_dot = lambda t, x : A2 @ x + b2.flatten()
        ts2, ys2, _ = simulate(x2_dot, T, x20, dt)

        """ 
            TODO: get accelerations from dynamics (use A1, b1, A2, b2???)
            - add acceleration for merged nodes
            - keep acceleration the same for non-merged nodes
        """
        ys

        def composite_system_dot(t, x):
            # Merge rows of A1, A2: Ac = merge(A1, A2) will merge rows, where merging means addition
            x1 = x[:4]
            x2 = x[2:]
            x1_dot = A1 @ x1 + b1.flatten()
            x2_dot = A2 @ x2 + b2.flatten()

            assert(x1_dot[2] == x2_dot[0])
            # TODO: what about the velocity?
            merged_node_vel = x1_dot[2]
            # The acceleration of merged nodes is the sum of accelerations of that node from all subsystems
            merged_node_acc = x1_dot[3] + x2_dot[1]
            # Recreate state-space dynamics
            x_c_dot = np.concatenate((x1_dot[:2], [merged_node_vel], [merged_node_acc], x2_dot[2:]))

            return x_c_dot

        ts_pred, ys_pred, _ = simulate(composite_system_dot, T, x0, dt)

        # Plot ys_pred vs ysc
        axes = plt.figure(layout="constrained", figsize=(10,20)).subplot_mosaic(
            """
            AA
            BB
            DE
            FF
            """
        )
        axes['A'].set_title('[1] Composite system: triple mass-spring')
        axes['A'].plot(tsc, ysc, label=labels)
        axes['A'].legend(loc="upper right")

        axes['B'].set_title('[2] Merged system')
        axes['B'].plot(ts_pred, ys_pred, label=labels)
        axes['B'].legend(loc="upper right")

        axes['D'].set_title('[3] Subsystem 1: double mass-spring')
        axes['D'].plot(ts1, ys1, label=labels[:4])
        axes['D'].legend(loc="upper right")

        axes['E'].set_title('[4] Subsystem 2: double free-spring')
        axes['E'].plot(ts2, ys2, label=labels[2:])
        axes['E'].legend(loc="upper right")

        axes['F'].set_title('[4] Errors: composite system - merged system')
        axes['F'].plot(tsc, ysc - ys_pred, label=labels)
        axes['F'].legend(loc="upper right")

        plt.savefig('test_merge.png')

    test_adding_edges()
    test_merging_nodes()