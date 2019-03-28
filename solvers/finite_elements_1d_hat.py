"""
 * Solution approximation using 1D hat functions.
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from core.parameters import a, f
from core.finite_elements import t_x, phi, phi_prime
from core.integrate import rectangle_midpoints

def assemble_A(N, P):
    A = np.zeros((N - 1, N - 1))
    for i in range(1, N):
        for j in range(1, N):
            h = lambda x : a(x) * phi_prime(i, x, N) * phi_prime(j, x, N)
            A[i - 1, j - 1] = rectangle_midpoints(h, t_x(i - 1, N), t_x(i + 1, N), N, P)
    return lil_matrix(A)

def assemble_B(N, P):
    B = np.zeros((N - 1, 1))
    for i in range(1, N):
        g = lambda x : phi(i, x, N) * f(x)
        B[i - 1] = rectangle_midpoints(g, t_x(i - 1, N), t_x(i + 1, N), N, P)
    return lil_matrix(B)

def assemble_U(N, P):
    A = assemble_A(N, P)
    B = assemble_B(N, P)
    U = spsolve(A, B)
    return U

def approximate_solution(x, U, N):
    '''
    Return the approximate solution taken in x.
    '''
    result = 0
    for i in range(N - 1):
        result += U[i] * phi(i + 1, x, N)
    return result


def approximate_derivative(x, U, N):
    '''
    Return the approximate derivative taken in x
    '''
    result = 0
    for i in range(N - 1):
        result += U[i] * phi_prime(i + 1, x, N)
    return result