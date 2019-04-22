"""
 * Solution approximation using 1D hat functions.
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from core.parameters import a, f
from core.finite_elements import t_x, phi_spline, phi_spline_prime, psi_spline, psi_spline_prime, tridiag
from core.integrate import rectangle_midpoints


def assemble_A_spline(N, P):
    A = csc_matrix((2 * N, 2 * N))

    A[:N-1, :N-1] = tridiag(1, N, t_x, a, phi_spline_prime, phi_spline_prime, N, P)

    for i in range(1, N):
        h = lambda x: a(x) * phi_spline_prime(i, x, N) * psi_spline_prime(i, x, N)
        A[i - 1, N - 1 + i] = rectangle_midpoints(h, t_x(i - 1, N), t_x(i + 1, N), N, P)

    for i in range(1, N):
        h = lambda x: a(x) * phi_spline_prime(i, x, N) * psi_spline_prime(i - 1, x, N)
        A[i - 1, N - 1 + i - 1] = rectangle_midpoints(h, t_x(i - 1, N), t_x(i + 1, N), N, P)

    for i in range(1, N):
        h = lambda x: a(x) * phi_spline_prime(i, x, N) * psi_spline_prime(i + 1, x, N)
        A[i - 1, N - 1 + i + 1] = rectangle_midpoints(h, t_x(i - 1, N), t_x(i + 1, N), N, P)

    # We symmetrize our matrix
    for i in range(1, N):
        A[N - 1 + i, i - 1] = A[i - 1, N - 1 + i]
        A[N - 1 + i - 1, i - 1] = A[i - 1, N - 1 + i - 1]
        A[N - 1 + i + 1, i - 1] = A[i - 1, N - 1 + i + 1]

    A[N-1:2*N, N-1:2*N] = tridiag(0, N+1, t_x, a, psi_spline_prime, psi_spline_prime, N, P)

    return A

def assemble_B_spline(N, P):
    B = np.zeros((2*N, 1))

    g = lambda x: phi_spline(1, x, N)
    first_int = rectangle_midpoints(g, 0, t_x(2, N), N, P)
    B[:N-1] = first_int*np.ones((N-1, 1))

    g = lambda x: psi_spline(1, x, N)
    second_int = rectangle_midpoints(g, 0, t_x(2, N), N, P)
    B[N-1:] = second_int*np.ones((N+1, 1))

    return B


def assemble_U_spline(N, P):
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