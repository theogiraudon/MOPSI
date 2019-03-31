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


# ----- Spline functions -----


# Four functions and their derivatives used in the definition of the spline functions

def g1(x):
    return -2 * x ** 3 + 3 * x ** 2


def g2(x):
    return 2 * x ** 3 - 3 * x ** 2 + 1


def g3(x):
    return x ** 3 - x ** 2


def g4(x):
    return x ** 3 - 2 * x ** 2 + x


def g1_prime(x):
    return -6 * x ** 2 + 6 * x


def g2_prime(x):
    return 6 * x ** 2 - 6 * x


def g3_prime(x):
    return 3 * x ** 2 - 2 * x


def g4_prime(x):
    return 3 * x ** 2 - 4 * x + 1


def phi_spline(i, x, N):
    """
        ith phi spline function using N nodes in the main mesh.

        The spline function support equates to [x_{i-1}, x_{i+1}], where x_i is the
        ith node of the main mesh.
    """
    if x < t_x(i + 1, N) and x >= t_x(i, N):
        return g2((x - t_x(i, N)) / (t_x(i + 1, N) - t_x(i, N)))
    elif x < t_x(i, N) and x >= t_x(i - 1, N):
        return g1((x - t_x(i - 1, N)) / (t_x(i, N) - t_x(i - 1, N)))
    else:
        return 0


def psi_spline(i, x, N):
    """
        ith psi spline function using N nodes in the main mesh.

        The spline function support equates to [x_{i-1}, x_{i+1}], where x_i is the
        ith node of the main mesh.
    """
    if x < t_x(i + 1, N) and x >= t_x(i, N):
        return g4((x - t_x(i, N)) / (t_x(i + 1, N) - t_x(i, N))) * (t_x(i + 1, N) - t_x(i, N))
    elif x < t_x(i, N) and x >= t_x(i - 1, N):
        return g3((x - t_x(i - 1, N)) / (t_x(i, N) - t_x(i - 1, N))) * (t_x(i, N) - t_x(i - 1, N))
    else:
        return 0


def phi_spline_prime(i, x, N):
    """
        Derivative of the ith phi spline function using N nodes in the main mesh.

        The spline function support equates to [x_{i-1}, x_{i+1}], where x_i is the
        ith node of the main mesh.
    """
    if x < t_x(i + 1, N) and x >= t_x(i, N):
        return g2_prime((x - t_x(i, N)) / (t_x(i + 1, N) - t_x(i, N))) / (t_x(i + 1, N) - t_x(i, N))
    elif x < t_x(i, N) and x >= t_x(i - 1, N):
        return g1_prime((x - t_x(i - 1, N)) / (t_x(i, N) - t_x(i - 1, N))) / (t_x(i, N) - t_x(i - 1, N))
    else:
        return 0


def psi_spline_prime(i, x, N):
    """
        Derivative of the ith psi spline function using N nodes in the main mesh.

        The spline function support equates to [x_{i-1}, x_{i+1}], where x_i is the
        ith node of the main mesh.
    """
    if x < t_x(i + 1, N) and x >= t_x(i, N):
        return g4_prime((x - t_x(i, N)) / (t_x(i + 1, N) - t_x(i, N)))
    elif x < t_x(i, N) and x >= t_x(i - 1, N):
        return g3_prime((x - t_x(i - 1, N)) / (t_x(i, N) - t_x(i - 1, N)))
    else:
        return 0