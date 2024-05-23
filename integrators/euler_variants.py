def euler(f, x, t, dt):
    return x + f(x, t) * dt

def semi_implicit_euler(f, q0, v0, t, dt):
    v = v0 + f([q0, v0], t) * dt
    q = q0 + v * dt
    return [q, v]