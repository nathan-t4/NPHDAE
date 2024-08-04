import numpy as np
from scipy.optimize import root

def radau_iia_2(fun, jac, t_span, y0, yp0, h):
    t = np.arange(t_span[0], t_span[1], h)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    # Radau IIA coefficients for s=2 stages (3rd-order method)
    A = np.array([
        [5/12, -1/12],
        [3/4, 1/4]
    ])
    b = np.array([3/4, 1/4])
    c = np.array([1/3, 1])

    for i in range(1, len(t)):
        t_n = t[i-1]
        y_n = y[i-1]

        def residual(Y):
            Y = Y.reshape(-1, 2)
            F = np.zeros_like(Y)
            for j in range(2):
                F[:, j] = fun(t_n + c[j]*h, Y[:, j], (Y[:, j] - y_n) / h)
            return Y.flatten() - np.tile(y_n, 2) - h * (A @ F.T).T.flatten()

        def jacobian(Y):
            Y = Y.reshape(-1, 2)
            J = np.zeros((2*len(y0), 2*len(y0)))
            for j in range(2):
                J[:, j*len(y0):(j+1)*len(y0)] = jac(t_n + c[j]*h, Y[:, j], (Y[:, j] - y_n) / h)
            return np.eye(2*len(y0)) - h * (np.kron(A, np.eye(len(y0))) @ J)

        Y_guess = np.tile(y_n, 2)
        sol = root(residual, Y_guess, jac=jacobian, method='hybr')
        Y = sol.x.reshape(-1, 2)

        y[i] = y_n + h * np.sum(b[j] * fun(t_n + c[j]*h, Y[:, j], (Y[:, j] - y_n) / h) for j in range(2))

    return t, y


# Example usage
if __name__ == "__main__":
    def fun(t, y, yp):
        return np.array([yp[0] - y[1], y[0]**2 + y[1]**2 - 1])

    def jac(t, y, yp):
        return np.array([
            [1, 0],
            [2*y[0], 2*y[1]]
        ])

    t_span = [0, 10]
    y0 = np.array([1, 0])
    yp0 = np.array([0, 1])
    h = 0.1

    t, y = radau_iia_2(fun, jac, t_span, y0, yp0, h)

    import matplotlib.pyplot as plt
    plt.plot(t, y[:, 0], label='y1')
    plt.plot(t, y[:, 1], label='y2')
    plt.legend()
    plt.xlabel('Time t')
    plt.ylabel('Solution y')
    plt.title('Radau IIA Method for Index-2 DAE')
    plt.show()