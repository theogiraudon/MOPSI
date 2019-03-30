"""
 * Galerkin decomposition methods (hat and spline functions).
"""

def t_x(i, N):
    '''
    ith node of the main mesh.
    '''
    return i / N

# ---- Hat functions ----

def phi(i, x, N):
    """
    ith hat function using N nodes in the main mesh.

    The hat function support equates to [x_{i-1}, x_{i+1}], where x_i is the
    ith node of the main mesh.
    """
    if t_x(i - 1, N) <= x < t_x(i, N):
        return (x - t_x(i - 1, N)) / (t_x(i, N) - t_x(i - 1, N))
    elif t_x(i, N) <= x < t_x(i + 1, N):
        return (t_x(i + 1, N) - x) / (t_x(i + 1, N) - t_x(i, N))
    else:
        return 0

def phi_prime(i, x, N):
    """
    Derivative of the ith hat function using N nodes in the main mesh.

    The hat function support equates to [x_{i-1}, x_{i+1}], where x_i is the
    ith node of the main mesh.
    """
    if t_x(i - 1, N) <= x < t_x(i, N):
        return 1 / (t_x(i, N) - t_x(i - 1, N))
    elif t_x(i, N) <= x < t_x(i + 1, N):
        return -1 / (t_x(i + 1, N) - t_x(i, N))
    else:
        return 0