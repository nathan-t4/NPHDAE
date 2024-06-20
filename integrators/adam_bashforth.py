def adam_bashforth(f, x, t, dt, T):
    # T can only be 1,2,3,4,5
    coeffs = [[1, 1],
              [2, 3, -1],
              [12, 23, -16, 5],
              [24, 55, -59, 37, -9],
              [720, 1901, -2774, 2616, -1274, 251]]
    x0 = x
    match T:
        case 1:
            return x0 + (dt / coeffs[T-1][0]) * (coeffs[T-1][1] * f(x0, t))
        case 2:
            x1 = x0 + (dt / coeffs[T-2][0]) * (coeffs[T-2][1] * f(x0, t))
            return x1 + (dt / coeffs[T-1][0]) * (coeffs[T-1][1] * f(x1, t) + coeffs[T-1][2] * f(x0, t))
        case 3:
            x1 = x0 + (dt / coeffs[T-3][0]) * (coeffs[T-3][1] * f(x0, t))
            x2 = x1 + (dt / coeffs[T-2][0]) * (coeffs[T-2][1] * f(x1, t) + coeffs[T-2][2] * f(x0, t))
            return x2 + (dt / coeffs[T-1][0]) * (coeffs[T-1][1] * f(x2, t) + coeffs[T-1][2] * f(x1, t) + coeffs[T-1][3] * f(x0, t))
        case 4:
            x1 = x0 + (dt / coeffs[T-4][0]) * (coeffs[T-4][1] * f(x0, t))
            x2 = x1 + (dt / coeffs[T-3][0]) * (coeffs[T-3][1] * f(x1, t) + coeffs[T-3][2] * f(x0, t))
            x3 = x2 + (dt / coeffs[T-2][0]) * (coeffs[T-2][1] * f(x2, t) + coeffs[T-2][2] * f(x1, t) + coeffs[T-2][3] * f(x0, t))
            return x3 + (dt / coeffs[T-1][0]) * (coeffs[T-1][1] * f(x3, t) + coeffs[T-1][2] * f(x2, t) + coeffs[T-1][3] * f(x1, t) + coeffs[T-1][4] * f(x0, t))
        case 5:
            x1 = x0 + (dt / coeffs[T-5][0]) * (coeffs[T-5][1] * f(x0, t))
            x2 = x1 + (dt / coeffs[T-4][0]) * (coeffs[T-4][1] * f(x1, t) + coeffs[T-4][2] * f(x0, t))
            x3 = x2 + (dt / coeffs[T-3][0]) * (coeffs[T-3][1] * f(x2, t) + coeffs[T-3][2] * f(x1, t) + coeffs[T-2][3] * f(x0, t))
            x4 = x3 + (dt / coeffs[T-2][0]) * (coeffs[T-2][1] * f(x3, t) + coeffs[T-2][2] * f(x2, t) + coeffs[T-2][3] * f(x1, t) + coeffs[T-2][4] * f(x0, t))
            return x4 + (dt / coeffs[T-1][0]) * (coeffs[T-1][1] * f(x4, t) + coeffs[T-1][2] * f(x3, t) + coeffs[T-1][3] * f(x2, t) + coeffs[T-1][4] * f(x1, t) + coeffs[T-1][5] * f(x0, t))
        case _:
            raise NotImplementedError(f'Adam-Bashforth method for {T} steps is not implemented yet!')