def euler(f, x, t, dt):
    return x + f(x, t) * dt

def semi_implicit_euler(f, q0, v0, t, dt):
    v = v0 + f([q0, v0], t) * dt
    q = q0 + v * dt
    return [q, v]

def sympletic_euler(f, x, t, dt):
    raise NotImplementedError # TODO
    p = x[p_idxs]
    q = x[q_idxs]
    next_p = p + dt * f(x, t)[q_idxs]
    new_x = []
    next_q = q + dt * f(new_x, t)[p_idxs]