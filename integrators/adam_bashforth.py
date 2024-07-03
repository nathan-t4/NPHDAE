def adam_bashforth(f, x, t, dt, T):
    # T can only be 1,2,3,4,5
    coeffs = [[1, 1],
              [2, 3, -1],
              [12, 23, -16, 5],
              [24, 55, -59, 37, -9],
              [720, 1901, -2774, 2616, -1274, 251]]
    x0 = x
    y0 = f(x0, t)
    match T:
        case 1:
            return x0 + (dt / coeffs[T-1][0]) * (coeffs[T-1][1] * y0)
        case 2:
            x1 = x0 + (dt / coeffs[T-2][0]) * (coeffs[T-2][1] * y0)
            return x1 + (dt / coeffs[T-1][0]) * (coeffs[T-1][1] * f(x1, t) + coeffs[T-1][2] * y0)
        case 3:
            x1 = x0 + (dt / coeffs[T-3][0]) * (coeffs[T-3][1] * y0)
            y1 = f(x1, t)
            x2 = x1 + (dt / coeffs[T-2][0]) * (coeffs[T-2][1] * y1 + coeffs[T-2][2] * y0)
            return x2 + (dt / coeffs[T-1][0]) * (coeffs[T-1][1] * f(x2, t) + coeffs[T-1][2] * y1 + coeffs[T-1][3] * y0)
        case 4:
            x1 = x0 + (dt / coeffs[T-4][0]) * (coeffs[T-4][1] * y0)
            y1 = f(x1, t)
            x2 = x1 + (dt / coeffs[T-3][0]) * (coeffs[T-3][1] * y1 + coeffs[T-3][2] * y0)
            y2 = f(x2, t)
            x3 = x2 + (dt / coeffs[T-2][0]) * (coeffs[T-2][1] * y2 + coeffs[T-2][2] * y1 + coeffs[T-2][3] * y0)
            return x3 + (dt / coeffs[T-1][0]) * (coeffs[T-1][1] * f(x3, t) + coeffs[T-1][2] * y2 + coeffs[T-1][3] * y1 + coeffs[T-1][4] * y0)
        case 5:
            x1 = x0 + (dt / coeffs[T-5][0]) * (coeffs[T-5][1] * y0)
            y1 = f(x1, t)
            x2 = x1 + (dt / coeffs[T-4][0]) * (coeffs[T-4][1] * y1 + coeffs[T-4][2] * y0)
            y2 = f(x2, t)
            x3 = x2 + (dt / coeffs[T-3][0]) * (coeffs[T-3][1] * y2 + coeffs[T-3][2] * y1 + coeffs[T-2][3] * y0)
            y3 = f(x3, t)
            x4 = x3 + (dt / coeffs[T-2][0]) * (coeffs[T-2][1] * y3 + coeffs[T-2][2] * y2 + coeffs[T-2][3] * y1 + coeffs[T-2][4] * y0)
            return x4 + (dt / coeffs[T-1][0]) * (coeffs[T-1][1] * f(x4, t) + coeffs[T-1][2] * y3 + coeffs[T-1][3] * y2 + coeffs[T-1][4] * y1 + coeffs[T-1][5] * y0)
        case _:
            raise NotImplementedError(f'Adam-Bashforth method for {T} steps is not implemented yet!')