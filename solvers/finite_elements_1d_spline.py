"""
 * Solution approximation using 1D hat functions.
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from core.parameters import a, f
from core.finite_elements import t_x, phi_spline, phi_spline_prime, psi_spline, psi_spline_prime
from core.integrate import rectangle_midpoints


def assemble_A_spline(N, P):
    A = np.zeros((2 * N, 2 * N))
    for i in range(1, N):
        for j in range(1, N):
            h = lambda x: a(x) * phi_spline_prime(i, x, N) * phi_spline_prime(j, x, N)
            A[i - 1, j - 1] = rectangle_midpoints(h, t_x(i - 1, N), t_x(i + 1, N), N, P)

    for i in range(1, N):
        for j in range(N + 1):
            h = lambda x: a(x) * phi_spline_prime(i, x, N) * psi_spline_prime(j, x, N)
            A[i - 1, N - 1 + j] = rectangle_midpoints(h, t_x(i - 1, N), t_x(i + 1, N), N, P)

    for i in range(N + 1):
        for j in range(1, N):
            h = lambda x: a(x) * psi_spline_prime(i, x, N) * phi_spline_prime(j, x, N)
            A[N - 1 + i, j - 1] = rectangle_midpoints(h, t_x(i - 1, N), t_x(i + 1, N), N, P)

    for i in range(N + 1):
        for j in range(N + 1):
            h = lambda x: a(x) * psi_spline_prime(i, x, N) * psi_spline_prime(j, x, N)
            A[N - 1 + i, N - 1 + j] = rectangle_midpoints(h, t_x(i - 1, N), t_x(i + 1, N), N, P)

    return A

def assemble_B_spline(N, P):
    B = np.zeros((2*N, 1))
    for i in range(1, N):
        g = lambda x: phi_spline(i, x, N) * f(x)
        B[i - 1] = rectangle_midpoints(g, t_x(i - 1, N), t_x(i + 1, N), N, P)
    for i in range(N + 1):
        g = lambda x: psi_spline(i, x, N) * f(x)
        B[N - 1 + i] = rectangle_midpoints(g, t_x(i - 1, N), t_x(i + 1, N), N, P)
    return B


def assemble_U_spline(N, P, R=1.):
    A = assemble_A_spline(N, P)
    B = assemble_B_spline(N, P)
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
        s += U[i - 1] * phi_spline_prime(i, x, N)
    for i in range(N + 1):
        s += U[N - 1 + i] * psi_spline_prime(i, x, N)
    return s