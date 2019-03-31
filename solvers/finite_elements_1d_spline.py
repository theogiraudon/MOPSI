"""
 * Solution approximation using 1D hat functions.
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from core.parameters import a, f
from core.finite_elements import t_x, phi, phi_prime, phi_spline, phi_spline_prime, psi_spline, psi_spline_prime
from core.integrate import rectangle_midpoints

def assemble_A_spline(N, P, R=1.):


    main_diagonal_up_left = [
        rectangle_midpoints(
            lambda x : a(x) * phi_spline_prime(i, x, N)**2,
            t_x(i - 1, N),
            t_x(i + 1, N),
            N,
            P
        )
        for i in range(1, N)
    ]
    upper_diagonal_up_left = [
        rectangle_midpoints(
            lambda x: a(x) * phi_spline_prime(i, x, N) * phi_spline_prime(i + 1, x, N),
            t_x(i - 1, N),
            t_x(i + 1, N),
            N,
            P
        )
        for i in range(1, N - 1)
    ]

    main_diagonal_up_right = [
        rectangle_midpoints(
            lambda x : a(x) * phi_spline_prime(i, x, N)*psi_spline_prime(i, x, N),
            t_x(i - 1, N),
            t_x(i + 1, N),
            N,
            P
        )
        for i in range(1, N)
    ]
    upper_diagonal_up_right = [
        rectangle_midpoints(
            lambda x: a(x) * phi_spline_prime(i, x, N) * psi_spline_prime(i + 1, x, N),
            t_x(i - 1, N),
            t_x(i + 1, N),
            N,
            P
        )
        for i in range(1, N + 1)
    ]

    main_diagonal_down_left = [
        rectangle_midpoints(
            lambda x : a(x) * psi_spline_prime(i, x, N)*phi_spline_prime(i, x, N),
            t_x(i - 1, N),
            t_x(i + 1, N),
            N,
            P
        )
        for i in range(N + 1)
    ]
    upper_diagonal_down_left = [
        rectangle_midpoints(
            lambda x: a(x) * psi_spline_prime(i, x, N) * phi_spline_prime(i + 1, x, N),
            t_x(i - 1, N),
            t_x(i + 1, N),
            N,
            P
        )
        for i in range(1, N - 1)
    ]

    main_diagonal_down_right = [
        rectangle_midpoints(
            lambda x : a(x) * phi_spline_prime(i, x, N)**2,
            t_x(i - 1, N),
            t_x(i + 1, N),
            N,
            P
        )
        for i in range(1, N)
    ]
    upper_diagonal_down_right = [
        rectangle_midpoints(
            lambda x: a(x) * psi_spline_prime(i, x, N) * psi_spline_prime(i + 1, x, N),
            t_x(i - 1, N),
            t_x(i + 1, N),
            N,
            P
        )
        for i in range(1, N - 1)
    ]

    A1 = diags([main_diagonal_up_left, upper_diagonal_up_left, upper_diagonal_up_left], [0, 1, -1], format="csc")
    A2 = diags([main_diagonal_up_right, upper_diagonal_up_right, upper_diagonal_up_right], [0, 1, -1], format="csc")
    A3 = diags([main_diagonal_down_left, upper_diagonal_down_left, upper_diagonal_down_left], [0, 1, -1], format="csc")
    A4 = diags([main_diagonal_down_right, upper_diagonal_down_right, upper_diagonal_down_right], [0, 1, -1], format="csc")

    return np.concatenate((np.concatenate((A1,A3)), np.concatenate((A2,A4))), axis=1)


def assemble_B_spline(N, P, R=1.):
    B = np.zeros((2*N, 1))
    for i in range(1, N):
        g = lambda x: rectangle_midpoints(i, x, N) * f(x)
        B[i - 1] = rectangle_midpoints(g, t_x(i - 1, N), t_x(i + 1, N), P)
    for i in range(N + 1):
        g = lambda x: psi_spline(i, x, N) * f(x)
        B[N - 1 + i] = rectangle_midpoints(g, t_x(i - 1, N), t_x(i + 1, N), P)
    return B


def assemble_U_spline(N, P, R=1.):
    A = assemble_A_spline(N, P, R)
    B = assemble_B_spline(N, P, R)
    U = spsolve(A, B)
    return U


def approximate_solution_spline(x, U, N):
    '''
    Returns the approximate solution taken in x using spline functions
    '''
    s = 0
    for i in range(1, N):
        s += U[i - 1] * phi_spline(i, x, N)
    for i in range(N + 1):
        s += U[N - 1 + i] * psi_spline(i, x, N)
    return s


def approximate_derivative_spline(x, U, N):
    '''
    Returns the approximate derivative taken x using spline functions
    '''
    s = 0
    for i in range(1, N):
        s += U[i - 1] * phi_prime(i, x, N)
    for i in range(N + 1):
        s += U[N - 1 + i] * psi_spline_prime(i, x, N)
    return s