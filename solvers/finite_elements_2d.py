import numpy as np
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve

from core.finite_elements import tridiag, tridiag2
from core.parameters import a_per, f, P, N_x, N_y
from core.finite_elements import t_x, t_y, phi, phi2D, phi2D_prime, phi_prime, psi2D, psi2D_prime, K
from core.integrate import rectangle_midpoints

def assemble_C(N_x, P):
    '''
       Assembling the 4 matrix C
    '''
    C = []
    C.append(tridiag(1, N_x + 1, t_x, lambda x: 1, phi_prime, phi_prime, N_x + 1, P))
    C.append(tridiag(1, N_x + 1, t_x, lambda x: 1 / x, phi, phi_prime, N_x + 1, P))
    C.append(tridiag(1, N_x + 1, t_x, lambda x: 1 / x, phi_prime, phi, N_x + 1, P))
    C.append(tridiag(1, N_x + 1, t_x, lambda x: 1 / x**2, phi, phi, N_x + 1, P))

    return C

def assemble_D(N_y, P):
    '''
       Assembling the 4 matrix D
    '''
    D = []
    D.append(tridiag2(0, N_y, t_y, a_per, psi2D, psi2D, N_y, P))
    D.append(tridiag2(0, N_y, t_y, a_per, psi2D_prime, psi2D, N_y, P))
    D.append(tridiag2(0, N_y, t_y, a_per, psi2D, psi2D_prime, N_y, P))
    D.append(tridiag2(0, N_y, t_y, a_per, psi2D_prime, psi2D_prime, N_y, P))

    return D

def assemble_F(N_x, N_y, P):
    '''
       Assembling the matrix F
    '''
    g = lambda x: phi(1, x, N_x + 1)
    h = lambda y: psi2D(0, y, N_y)

    int_phi = rectangle_midpoints(g, t_x(0, N_x + 1), t_x(2, N_x + 1), N_x + 1, P)
    int_psi = rectangle_midpoints(h, t_y(0, N_y), t_y(2, N_y), N_y, P)

    return (int_phi*int_psi)*np.ones((N_x * N_y, 1))

def assemble_B(C, D, N_x, N_y):
    '''
       Assembling the matrix B
    '''
    B = np.zeros((N_x * N_y, N_x * N_y))

    for i in range(N_x):
        for j in range(N_y):
            B[K(i, j, N_y), K(i, j, N_y)] = sum([C[m][i, i] * D[m][j, j] for m in range(4)])
    for i in range(N_x - 1):
        for j in range(N_y):
            B[K(i, j, N_y), K(i + 1, j, N_y)] = sum([C[m][i, i + 1] * D[m][j, j] for m in range(4)])
    for i in range(N_x - 1):
        for j in range(N_y):
            B[K(i + 1, j, N_y), K(i, j, N_y)] = sum([C[m][i + 1, i] * D[m][j, j] for m in range(4)])

    for i in range(N_x):
        for j in range(N_y - 1):
            B[K(i, j, N_y), K(i, j + 1, N_y)] = sum([C[m][i, i] * D[m][j, j + 1] for m in range(4)])
    for i in range(N_x - 1):
        for j in range(N_y - 1):
            B[K(i, j, N_y), K(i + 1, j + 1, N_y)] = sum([C[m][i, i + 1] * D[m][j, j + 1] for m in range(4)])
    for i in range(N_x - 1):
        for j in range(N_y - 1):
            B[K(i + 1, j, N_y), K(i, j + 1, N_y)] = sum([C[m][i + 1, i] * D[m][j, j + 1] for m in range(4)])

    for i in range(N_x):
        for j in range(N_y - 1):
            B[K(i, j + 1, N_y), K(i, j, N_y)] = sum([C[m][i, i] * D[m][j + 1, j] for m in range(4)])
    for i in range(N_x - 1):
        for j in range(N_y - 1):
            B[K(i, j + 1, N_y), K(i + 1, j, N_y)] = sum([C[m][i, i + 1] * D[m][j + 1, j] for m in range(4)])
    for i in range(N_x - 1):
        for j in range(N_y - 1):
            B[K(i + 1, j + 1, N_y), K(i, j, N_y)] = sum([C[m][i + 1, i] * D[m][j + 1, j] for m in range(4)])

    for i in range(N_x):
        B[K(i, 0, N_y), K(i, N_y - 1, N_y)] = sum([C[m][i, i] * D[m][0, N_y - 1] for m in range(4)])
        B[K(i, N_y - 1, N_y), K(i, 0, N_y)] = sum([C[m][i, i] * D[m][N_y - 1, 0] for m in range(4)])
    for i in range(N_x - 1):
        B[K(i + 1, 0, N_y), K(i, N_y - 1, N_y)] = sum([C[m][i + 1, i] * D[m][0, N_y - 1] for m in range(4)])
        B[K(i + 1, N_y - 1, N_y), K(i, 0, N_y)] = sum([C[m][i + 1, i] * D[m][N_y - 1, 0] for m in range(4)])
    for i in range(N_x - 1):
        B[K(i, 0, N_y), K(i + 1, N_y - 1, N_y)] = sum([C[m][i, i + 1] * D[m][0, N_y - 1] for m in range(4)])
        B[K(i, N_y - 1, N_y), K(i + 1, 0, N_y)] = sum([C[m][i, i + 1] * D[m][N_y - 1, 0] for m in range(4)])

    return B

def assemble_U(N_x, N_y, P):
    C = assemble_C(N_x, P)
    D = assemble_D(N_y, P)
    B = assemble_B(C, D, N_x, N_y)
    F = assemble_F(N_x, N_y, P)
    B = csc_matrix(B)
    return spsolve(B, F)

def approximate_solution_2D(x, y, U, N_x, N_y):
    s=0
    # Periodicity of y
    y = y - np.floor(y)
    for i in range(N_x):
        for j in range(N_y):
            s += U[K(i, j, N_y)] * phi(i + 1, x, N_x + 1) * psi2D(j, y, N_y)
    return s

def approximate_derivative_2D(x, y, U, N_x, N_y):
    s=0
    # Periodicity of y
    y = y - np.floor(y)
    s = sum( [ (U[K(i, j, N_y)] * (phi_prime(i + 1, x, N_x + 1) * psi2D(j, y, N_y)
                     + (1.0 / x) * phi(i + 1, x, N_x + 1) * psi2D_prime(j, y, N_y))) for i in range(N_x) for j in range(N_y) ] )
    return s
