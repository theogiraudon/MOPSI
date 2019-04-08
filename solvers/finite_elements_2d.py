import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from core.finite_elements import tridiag, tridiag2
from core.parameters import a_per, f, P
from core.finite_elements import t_x, t_y, phi, phi2D, phi2D_prime, phi_prime, psi2D, psi2D_prime, K
from core.integrate import rectangle_midpoints

def assemble_C_2D(N_x, P):
    '''
       Assembling the 4 matrix C
    '''
    C = []
    C.append(tridiag(1, N_x + 1, t_x, lambda x: 1, phi_prime, phi_prime, N_x + 1, P))
    C.append(tridiag(1, N_x + 1, t_x, lambda x: 1 / x, phi, phi_prime, N_x + 1, P))
    C.append(tridiag(1, N_x + 1, t_x, lambda x: 1 / x, phi_prime, phi, N_x + 1, P))
    C.append(tridiag(1, N_x + 1, t_x, lambda x: 1 / x**2, phi, phi, N_x + 1, P))

    return C

def assemble_D_2D(N_y, P):
    '''
       Assembling the 4 matrix D
    '''
    D = []
    D.append(tridiag2(0, N_y, t_y, a_per, psi2D, psi2D, N_y, P))
    D.append(tridiag2(0, N_y, t_y, a_per, psi2D_prime, psi2D, N_y, P))
    D.append(tridiag2(0, N_y, t_y, a_per, psi2D, psi2D_prime, N_y, P))
    D.append(tridiag2(0, N_y, t_y, a_per, psi2D_prime, psi2D_prime, N_y, P))
    print(D[0])

    return D

def assemble_F_2D(N_x, N_y, P):
    '''
       Assembling the matrix F
    '''
    F = np.zeros(N_x * N_y)
    for i in range(0, N_x):
        for j in range(0, N_y):
            g = lambda x: phi2D(i, x, N_x) * f(x)
            h = lambda y: psi2D(j, y, N_y)

            result_g = rectangle_midpoints(g, t_x(i, N_x), t_x(i + 2, N_x), N_x, P)
            if j == (N_y - 1):
                result_h = (rectangle_midpoints(h, 0, t_y(1, N_y), N_y, P) +
                            rectangle_midpoints(h, t_y(N_y - 1, N_y), 1, N_y, P))
            else:
                result_h = rectangle_midpoints(h, t_y(j, N_y), t_y(j + 2, N_y), N_y, P)
            F[K(i, j, N_y)] = result_g * result_h
    return F

def assemble_B_2D(C, D, N_x, N_y):
    '''
       Assembling the matrix B
    '''
    B = np.zeros((N_x * N_y, N_x * N_y))

    for i in range(0, N_x):
        for j in range(0, N_y):
            for k in range(0, N_x):
                for l in range(0, N_y):
                    B[K(i, j, N_y), K(k, l, N_y)] = sum([C[m][i, k] * D[m][j, l] for m in range(4)])
    return B

def assemble_U_2D(N_x, N_y, P):
    C = assemble_C_2D(N_x, P)
    D = assemble_D_2D(N_y, P)
    B = assemble_B_2D(C, D, N_x, N_y)
    F = assemble_F_2D(N_x, N_y, P)
    return spsolve(B, F)

def approximate_solution_2D(x, y, U, N_x, N_y):
    s=0
    # Periodicity of y
    y = y - np.floor(y)
    for i in range(0, N_x):
        for j in range(0, N_y):
            s += U[K(i, j, N_y)] * phi2D(i, x, N_x) * psi2D(j, y, N_y)
    return s

def approximate_derivative_2D(x, y, U, N_x, N_y):
    s=0
    # Periodicity of y
    y = y - np.floor(y)
    for i in range(0, N_x):
        for j in range(0, N_y):
            s += (U[K(i, j, N_y)]
                  * (phi2D_prime(i, x, N_x) * psi2D(j, y, N_y)
                     + (1.0 / x) * phi2D(i, x, N_x) * psi2D_prime(j, y, N_y)))
    return s
