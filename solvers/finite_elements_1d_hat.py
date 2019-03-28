"""
 * Solution approximation using 1D hat functions.
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from core.parameters import a, f
from core.finite_elements import t_x, phi, phi_prime
from core.integrate import rectangle_midpoints

def assemble_A(N, P):
    main_diagonal = [
        rectangle_midpoints(
            lambda x : a(x) * phi_prime(i, x, N) ** 2,
            t_x(i - 1, N),
            t_x(i + 1, N),
            N,
            P
        )
        for i in range(1, N)
    ]
    upper_diagonal = [
        rectangle_midpoints(
            lambda x: a(x) * phi_prime(i, x, N) * phi_prime(i + 1, x, N),
            t_x(i - 1, N),
            t_x(i + 1, N),
            N,
            P
        )
        for i in range(1, N - 1)
    ]
    return diags([main_diagonal, upper_diagonal, upper_diagonal], [0, 1, -1], format="csc") # A is symmetric.

def assemble_B(N, P):
    B = np.zeros((N - 1, 1))
    for i in range(1, N):
        g = lambda x : phi(i, x, N) * f(x)
        B[i - 1] = rectangle_midpoints(g, t_x(i - 1, N), t_x(i + 1, N), N, P)
    return B

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