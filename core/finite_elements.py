from core.integrate import rectangle_midpoints
from scipy.sparse import diags
import numpy as np
import matplotlib.pyplot as plt
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
    if i > 0 and i < N:
        if x < t_x(i + 1, N) and x >= t_x(i, N):
            return g4((x - t_x(i, N)) / (t_x(i + 1, N) - t_x(i, N))) * (t_x(i + 1, N) - t_x(i, N))
        elif x < t_x(i, N) and x >= t_x(i - 1, N):
            return g3((x - t_x(i - 1, N)) / (t_x(i, N) - t_x(i - 1, N))) * (t_x(i, N) - t_x(i - 1, N))
        else:
            return 0
    elif i == 0:
        if x < t_x(1, N) and x >= 0:
            return g4(x / t_x(1, N)) * t_x(1, N)
        else:
            return 0
    elif i == N:
        if x < 1 and x >= t_x(N - 1, N):
            return g3((x - t_x(N - 1, N)) / (1 - t_x(N - 1, N))) * (1 - t_x(N - 1, N))
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
    if i > 0 and i < N:
        if x < t_x(i + 1, N) and x >= t_x(i, N):
            return g4_prime((x - t_x(i, N)) / (t_x(i + 1, N) - t_x(i, N)))
        elif x < t_x(i, N) and x >= t_x(i - 1, N):
            return g3_prime((x - t_x(i - 1, N)) / (t_x(i, N) - t_x(i - 1, N)))
        else:
            return 0
    elif i == 0:
        if x < t_x(1, N) and x >= 0:
            return g4_prime(x / t_x(1, N))
        else:
            return 0
    elif i == N:
        if x <= 1 and x >= t_x(N - 1, N):
            return g3_prime((x - t_x(N - 1, N)) / (1 - t_x(N - 1, N)))
        else:
            return 0



"""
     Galerkin decomposition methods (2D hat functions)
"""

def t_y(j, N_y):
    return j / N_y

def K(i, j, N_y):
    return i*N_y + j


# ---- fonctions chapeaux--------
def phi2D(i, x, N_x):  # Defines phi_0 to phi_{N_x - 1}
    if x < t_x(i + 2, N_x) and x >= t_x(i + 1, N_x):
        return (t_x(i + 2, N_x) - x) / (t_x(i + 2, N_x) - t_x(i + 1, N_x))
    elif x < t_x(i + 1, N_x) and x >= t_x(i, N_x):
        return (x - t_x(i, N_x)) / (t_x(i + 1, N_x) - t_x(i, N_x))
    else:
        return 0


def phi2D_prime(i, x, N_x):  # Defines phi'_0 to phi'_{N_x - 1}
    if x < t_x(i + 2, N_x) and x >= t_x(i + 1, N_x):
        return -1. / (t_x(i + 2, N_x) - t_x(i + 1, N_x))
    elif x < t_x(i + 1, N_x) and x >= t_x(i, N_x):
        return 1. / (t_x(i + 1, N_x) - t_x(i, N_x))
    else:
        return 0


def psi2D(j, y, N_y):  # Defines psi_0 to psi_{N_y - 1}
    if j < (N_y - 1):
        if y < t_y(j + 2, N_y) and y >= t_y(j + 1, N_y):
            return (t_y(j + 2, N_y) - y) / (t_y(j + 2, N_y) - t_y(j + 1, N_y))
        elif y < t_y(j + 1, N_y) and y >= t_y(j, N_y):
            return (y - t_y(j, N_y)) / (t_y(j + 1, N_y) - t_y(j, N_y))
        else:
            return 0
    else:
        if y < t_y(1, N_y):
            return (t_y(1, N_y) - y) / (t_y(1, N_y))
        elif y >= t_y(N_y - 1, N_y):
            return (y - t_y(N_y - 1, N_y)) / (1 - t_y(N_y - 1, N_y))
        else:
            return 0


def psi2D_prime(j, y, N_y):  # Defines psi'_0 to psi'_{N_y - 1}
    if j < (N_y - 1):
        if y < t_y(j + 2, N_y) and y >= t_y(j + 1, N_y):
            return -1. / (t_y(j + 2, N_y) - t_y(j + 1, N_y))
        elif y < t_y(j + 1, N_y) and y >= t_y(j, N_y):
            return 1. / (t_y(j + 1, N_y) - t_y(j, N_y))
        else:
            return 0
    else:
        if y < t_y(1, N_y):
            return -1 / (t_y(1, N_y) - t_y(0, N_y))
        elif y >= t_y(N_y - 1, N_y):
            return 1 / (t_y(j, N_y) - t_y(j - 1, N_y))
        else:
            return 0


def tridiag(begin, end, t, g, h1, h2, N, P):
    main_diagonal = [
        rectangle_midpoints(
            lambda x: g(x) * h1(i, x, N) * h2(i, x, N),
            t(i - 1, N),
            t(i + 1, N),
            N,
            P
        )
        for i in range(begin, end)
    ]

    upper_diagonal = [
        rectangle_midpoints(
            lambda x: g(x) * h1(i, x, N) * h2(i + 1, x, N),
            t(i, N),
            t(i + 1, N),
            N,
            P
        )
        for i in range(begin, end - 1)
    ]

    bottom_diagonal = [
        rectangle_midpoints(
            lambda x: g(x) * h2(i, x, N) * h1(i + 1, x, N),
            t(i, N),
            t(i + 1, N),
            N,
            P
        )
        for i in range(begin, end - 1)
    ]

    return diags([main_diagonal, upper_diagonal, bottom_diagonal], [0, 1, -1], format="csc")


def tridiag2(begin, end, t, g, h1, h2, N, P):

    main_diagonal = [
        rectangle_midpoints(
            lambda x: g(x) * h1(i, x, N) * h2(i, x, N),
            t(i, N),
            t(i + 2, N),
            N,
            P
        )
        for i in range(begin, end - 1)
    ]

    main_diagonal.append(rectangle_midpoints(
            lambda x: g(x) * h1(N - 1, x, N) * h2(N - 1, x, N),
            t(N - 1, N),
            t(N, N),
            N,
            P
        )\
               + rectangle_midpoints(
        lambda x: g(x) * h1(N - 1, x, N) * h2(N - 1, x, N),
        t(0, N),
        t(1, N),
        N,
        P
    ))

    upper_diagonal = [
        rectangle_midpoints(
            lambda x: g(x) * h1(i, x, N) * h2(i + 1, x, N),
            t(i + 1, N),
            t(i + 2, N),
            N,
            P
        )
        for i in range(begin, end - 1)
    ]

    bottom_diagonal = [
        rectangle_midpoints(
            lambda x: g(x) * h2(i, x, N) * h1(i + 1, x, N),
            t(i + 1, N),
            t(i + 2, N),
            N,
            P
        )
        for i in range(begin, end - 1)
    ]

    M = diags([main_diagonal, upper_diagonal, bottom_diagonal], [0, 1, -1], format="csc")

    M[0, -1] = rectangle_midpoints(
            lambda x: g(x) * h1(0, x, N) * h2(N - 1, x, N),
            t(0, N),
            t(1, N),
            N,
            P
        )

    M[-1, 0] = rectangle_midpoints(
        lambda x: g(x) * h1(N - 1, x, N) * h2(0, x, N),
        t(0, N),
        t(1, N),
        N,
        P
    )

    return M


def tridiag3(begin, end, t, g, h1, h2, N, P):

    main_diagonal = [
        rectangle_midpoints(
            lambda x: g(x) * h1(i, x, N) * h2(i, x, N),
            t(i - 1, N),
            t(i + 1, N),
            N,
            P
        )
        for i in range(begin, end - 1)
    ]

    main_diagonal.append(rectangle_midpoints(
            lambda x: g(x) * h1(N, x, N) * h2(N, x, N),
            t(N - 1, N),
            t(N, N),
            N,
            P
        )\
               + rectangle_midpoints(
        lambda x: g(x) * h1(N, x, N) * h2(N, x, N),
        t(0, N),
        t(1, N),
        N,
        P
    ))

    upper_diagonal = [
        rectangle_midpoints(
            lambda x: g(x) * h1(i, x, N) * h2(i + 1, x, N),
            t(i, N),
            t(i + 1, N),
            N,
            P
        )
        for i in range(begin, end - 1)
    ]

    bottom_diagonal = [
        rectangle_midpoints(
            lambda x: g(x) * h2(i, x, N) * h1(i + 1, x, N),
            t(i, N),
            t(i + 1, N),
            N,
            P
        )
        for i in range(begin, end - 1)
    ]

    M = diags([main_diagonal, upper_diagonal, bottom_diagonal], [0, 1, -1], format="csc")

    M[0, -1] = rectangle_midpoints(
            lambda x: g(x) * h1(1, x, N) * h2(N, x, N),
            t(0, N),
            t(1, N),
            N,
            P
        )

    M[-1, 0] = rectangle_midpoints(
        lambda x: g(x) * h1(N, x, N) * h2(1, x, N),
        t(0, N),
        t(1, N),
        N,
        P
    )

    return M