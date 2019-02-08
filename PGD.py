# -*- coding: utf-8 -*-
"""
Created on Tue Feb 5 13:40:19 2019

@author: Lauret Jérémy
"""

# ---- importations de librairies ------
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
from scipy.sparse.linalg import spsolve
from scipy import sparse
import math

np.set_printoptions(precision=4)  # pour joli affichage des matrices

# --------------------------------
#
#     PARAMETRES DU CALCUL
#
# --------------------------------
step = 1.              # Roughly, step of the approximation.
N_x = 30               # Dimension of the x range approximation space.
N_y = 31               # Dimension of the y range approximation space.
nb_iter = 30           # Number of PGD iterations
eps = 1e-16            # Fixed point algorithm precision
max_rand_int = 1000    # Random coefficients picked for the PGD initialization belong to [-max_rand_int, max_rand_int]


def t_x(i, N_x, step=1.):
    return i * step / (N_x + 1)

def t_y(j, N_y, step=1.):
    return j * step / N_y

def a_per(y):
    return 1 + np.sin(math.pi * y) ** 2
    #return 2 + np.cos(2*math.pi*y)
    #return 1

def a(x):
    return a_per(np.log(x))

def f(x):
    return 1.

def K(i, j, N_y):
    return i * N_y + j

# ---- fonctions chapeaux--------
def phi(i, x, N_x, step=1.):  # Defines phi_0 to phi_{N_x - 1}
    if x < t_x(i + 2, N_x, step) and x >= t_x(i + 1, N_x, step):
        return (t_x(i + 2, N_x, step) - x) / (t_x(i + 2, N_x, step) - t_x(i + 1, N_x, step))
    elif x < t_x(i + 1, N_x, step) and x >= t_x(i, N_x, step):
        return (x - t_x(i, N_x, step)) / (t_x(i + 1, N_x, step) - t_x(i, N_x, step))
    else:
        return 0

def phi_prime(i, x, N_x, step=1.):  # Defines phi'_0 to phi'_{N_x - 1}
    if x < t_x(i + 2, N_x, step) and x >= t_x(i + 1, N_x, step):
        return -1. / (t_x(i + 2, N_x, step) - t_x(i + 1, N_x, step))
    elif x < t_x(i + 1, N_x, step) and x >= t_x(i, N_x, step):
        return 1. / (t_x(i + 1, N_x, step) - t_x(i, N_x, step))
    else:
        return 0

def psi(j, y, N_y, step=1.):  # Defines psi_0 to psi_{N_y - 1}
    if j < (N_y - 1):
        if y < t_y(j + 2, N_y, step) and y >= t_y(j + 1, N_y, step):
            return (t_y(j + 2, N_y, step) - y) / (t_y(j + 2, N_y, step) - t_y(j + 1, N_y, step))
        elif y < t_y(j + 1, N_y, step) and y >= t_y(j, N_y, step):
            return (y - t_y(j, N_y, step)) / (t_y(j + 1, N_y, step) - t_y(j, N_y, step))
        else:
            return 0
    else:
        if y < t_y(1, N_y, step):
            return (t_y(1, N_y, step) - y) / (t_y(1, N_y, step) - t_y(0, N_y, step))
        elif y >= t_y(N_y - 1, N_y, step):
            return (y - t_y(N_y - 1, N_y, step)) / (t_y(N_y, N_y, step) - t_y(N_y - 1, N_y, step))
        else:
            return 0

def psi_prime(j, y, N_y, step=1.):  # Defines psi'_0 to psi'_{N_y - 1}
    if j < (N_y - 1):
        if y < t_y(j + 2, N_y, step) and y >= t_y(j + 1, N_y, step):
            return -1. / (t_y(j + 2, N_y, step) - t_y(j + 1, N_y, step))
        elif y < t_y(j + 1, N_y, step) and y >= t_y(j, N_y, step):
            return 1. / (t_y(j + 1, N_y, step) - t_y(j, N_y, step))
        else:
            return 0
    else:
        if y < t_y(1, N_y, step):
            return -1 / (t_y(1, N_y, step) - t_y(0, N_y, step))
        elif y >= t_y(N_y - 1, N_y, step):
            return 1 / (t_y(j, N_y, step) - t_y(j - 1, N_y, step))
        else:
            return 0

'''
X = np.linspace(0, 1, 5000)
Y = [[psi(k, x, N_y) for x in X] for k in range(N_y)]
for k in range(N_y):
    plt.plot(X, Y[k])
plt.show()
print("hello")
'''

# ------------------- Assemble iteration-wise constant matrix -----------------------

def assemble_C(N_x, step=1.):
    '''
       Assemblage des matrices C.
       Équivalence PGD : C[0] = D, C[1] = Mat(int(1/x * phi_i * phi_j')),
                         C[2] = Mat(int(1/x * phi_i' * phi_j)), C[3] = M_{1/x^2}
    '''
    C = [np.zeros((N_x, N_x)) for k in range(4)]

    for i in range(0, N_x):
        for k in range(0, N_x):
            h = lambda x: phi_prime(i, x, N_x, step) * phi_prime(k, x, N_x, step)
            result_h = scipy.integrate.quad(h, t_x(i, N_x, step), t_x(i + 2, N_x, step), epsrel=1e-16)
            C[0][i][k] = result_h[0]

    for i in range(0, N_x):
        for k in range(0, N_x):
            h = lambda x: (1 / x) * phi(i, x, N_x, step) * phi_prime(k, x, N_x, step)
            result_h = scipy.integrate.quad(h, t_x(i, N_x, step), t_x(i + 2, N_x, step), epsrel=1e-16)
            C[1][i][k] = result_h[0]

    for i in range(0, N_x):
        for k in range(0, N_x):
            h = lambda x: (1 / x) * phi_prime(i, x, N_x, step) * phi(k, x, N_x, step)
            result_h = scipy.integrate.quad(h, t_x(i, N_x, step), t_x(i + 2, N_x, step), epsrel=1e-16)
            C[2][i][k] = result_h[0]

    for i in range(0, N_x):
        for k in range(0, N_x):
            h = lambda x: (1 / x ** 2) * phi(i, x, N_x, step) * phi(k, x, N_x, step)
            result_h = scipy.integrate.quad(h, t_x(i, N_x, step), t_x(i + 2, N_x, step), epsrel=1e-16)
            C[3][i][k] = result_h[0]
    return C

def assemble_D(N_y, step=1.):
    '''
       Assemblage des matrices D.
       Équivalence PGD : D[0] = M_aper, D[1] = Mat(int(aper * psi_i' * psi_j)),
                         D[2] = Mat(int(aper * psi_i * psi_j')), D[3] = D_aper
    '''
    D = [np.zeros((N_y, N_y)) for k in range(4)]

    for j in range(0, N_y):
        for l in range(0, N_y):
            h = lambda y: a_per(y) * psi(j, y, N_y, step) * psi(l, y, N_y, step)
            if j == (N_y - 1):
                result_h = (scipy.integrate.quad(h, 0, t_y(1, N_y, step), epsrel=1e-16)[0] +
                            scipy.integrate.quad(h, t_y(N_y - 1, N_y, step), 1, epsrel=1e-16)[0])
            else:
                result_h = scipy.integrate.quad(h, t_y(j, N_y, step), t_y(j + 2, N_y, step), epsrel=1e-16)[0]
            D[0][j][l] = result_h

    for j in range(0, N_y):
        for l in range(0, N_y):
            h = lambda y: a_per(y) * psi_prime(j, y, N_y, step) * psi(l, y, N_y, step)
            if j == (N_y - 1):
                result_h = (scipy.integrate.quad(h, 0, t_y(1, N_y, step), epsrel=1e-16)[0] +
                            scipy.integrate.quad(h, t_y(N_y - 1, N_y, step), 1, epsrel=1e-16)[0])
            else:
                result_h = scipy.integrate.quad(h, t_y(j, N_y, step), t_y(j + 2, N_y, step), epsrel=1e-16)[0]
            D[1][j][l] = result_h

    for j in range(0, N_y):
        for l in range(0, N_y):
            h = lambda y: a_per(y) * psi(j, y, N_y, step) * psi_prime(l, y, N_y, step)
            if j == (N_y - 1):
                result_h = (scipy.integrate.quad(h, 0, t_y(1, N_y, step), epsrel=1e-16)[0] +
                            scipy.integrate.quad(h, t_y(N_y - 1, N_y, step), 1, epsrel=1e-16)[0])
            else:
                result_h = scipy.integrate.quad(h, t_y(j, N_y, step), t_y(j + 2, N_y, step), epsrel=1e-16)[0]
            D[2][j][l] = result_h

    for j in range(0, N_y):
        for l in range(0, N_y):
            h = lambda y: a_per(y) * psi_prime(j, y, N_y, step) * psi_prime(l, y, N_y, step)
            if j == (N_y - 1):
                result_h = (scipy.integrate.quad(h, 0, t_y(1, N_y, step), epsrel=1e-16)[0] +
                            scipy.integrate.quad(h, t_y(N_y - 1, N_y, step), 1, epsrel=1e-16)[0])
            else:
                result_h = scipy.integrate.quad(h, t_y(j, N_y, step), t_y(j + 2, N_y, step), epsrel=1e-16)[0]
            D[3][j][l] = result_h
    return D

def assemble_F(N_x, N_y, R=1.):
    '''
       Assemblage de la matrice F
    '''
    F = np.zeros(N_x * N_y)
    for i in range(0, N_x):
        for j in range(0, N_y):
            g = lambda x: phi(i, x, N_x, R) * f(x)
            h = lambda y: psi(j, y, N_y, R)

            result_g = scipy.integrate.quad(g, t_x(i, N_x, R), t_x(i + 2, N_x, R), epsrel=1e-16)[0]
            if j == (N_y - 1):
                result_h = (scipy.integrate.quad(h, 0, t_y(1, N_y, R), epsrel=1e-16)[0] +
                            scipy.integrate.quad(h, t_y(N_y - 1, N_y, R), 1, epsrel=1e-16)[0])
            else:
                result_h = scipy.integrate.quad(h, t_y(j, N_y, R), t_y(j + 2, N_y, R), epsrel=1e-16)[0]
            F[K(i, j, N_y)] = result_g * result_h
    return F

def assemble_E(N_y, step=1.):
    '''
       Assemblage des matrices E.
       Équivalence PGD : E[0] = M_psi, E[1] = Mat(int(psi_i' * psi_j)),
                         E[2] = Mat(int(psi_i * psi_j')), E[3] = D_psi
    '''
    E = [np.zeros((N_y, N_y)) for k in range(4)]

    for j in range(0, N_y):
        for l in range(0, N_y):
            h = lambda y: psi(j, y, N_y, step) * psi(l, y, N_y, step)
            if j == (N_y - 1):
                result_h = (scipy.integrate.quad(h, 0, t_y(1, N_y, step), epsrel=1e-16)[0] +
                            scipy.integrate.quad(h, t_y(N_y - 1, N_y, step), 1, epsrel=1e-16)[0])
            else:
                result_h = scipy.integrate.quad(h, t_y(j, N_y, step), t_y(j + 2, N_y, step), epsrel=1e-16)[0]
            E[0][j][l] = result_h

    for j in range(0, N_y):
        for l in range(0, N_y):
            h = lambda y: psi_prime(j, y, N_y, step) * psi(l, y, N_y, step)
            if j == (N_y - 1):
                result_h = (scipy.integrate.quad(h, 0, t_y(1, N_y, step), epsrel=1e-16)[0] +
                            scipy.integrate.quad(h, t_y(N_y - 1, N_y, step), 1, epsrel=1e-16)[0])
            else:
                result_h = scipy.integrate.quad(h, t_y(j, N_y, step), t_y(j + 2, N_y, step), epsrel=1e-16)[0]
            E[1][j][l] = result_h

    for j in range(0, N_y):
        for l in range(0, N_y):
            h = lambda y: psi(j, y, N_y, step) * psi_prime(l, y, N_y, step)
            if j == (N_y - 1):
                result_h = (scipy.integrate.quad(h, 0, t_y(1, N_y, step), epsrel=1e-16)[0] +
                            scipy.integrate.quad(h, t_y(N_y - 1, N_y, step), 1, epsrel=1e-16)[0])
            else:
                result_h = scipy.integrate.quad(h, t_y(j, N_y, step), t_y(j + 2, N_y, step), epsrel=1e-16)[0]
            E[2][j][l] = result_h

    for j in range(0, N_y):
        for l in range(0, N_y):
            h = lambda y: psi_prime(j, y, N_y, step) * psi_prime(l, y, N_y, step)
            if j == (N_y - 1):
                result_h = (scipy.integrate.quad(h, 0, t_y(1, N_y, step), epsrel=1e-16)[0] +
                            scipy.integrate.quad(h, t_y(N_y - 1, N_y, step), 1, epsrel=1e-16)[0])
            else:
                result_h = scipy.integrate.quad(h, t_y(j, N_y, step), t_y(j + 2, N_y, step), epsrel=1e-16)[0]
            E[3][j][l] = result_h
    return E

def assemble_F_1(N_x, step=1.):
    '''
       Assemblage de la matrice F_1.
    '''
    F_1 = np.zeros(N_x)
    for i in range(0, N_x):
        g = lambda x: phi(i, x, N_x, step) * f(x)
        result_g = scipy.integrate.quad(g, t_x(i, N_x, step), t_x(i + 2, N_x, step), epsrel=1e-16)[0]
        F_1[i] = result_g
    return F_1

def assemble_F_2(N_y, step=1.):
    '''
       Assemblage de la matrice F_2.
    '''
    F_2 = np.zeros(N_y)
    for j in range(0, N_y):
        h = lambda y: psi(j, y, N_y, step)
        if j == (N_y - 1):
            result_h = (scipy.integrate.quad(h, 0, t_y(1, N_y, step), epsrel=1e-16)[0] +
                        scipy.integrate.quad(h, t_y(N_y - 1, N_y, step), 1, epsrel=1e-16)[0])
        else:
            result_h = scipy.integrate.quad(h, t_y(j, N_y, step), t_y(j + 2, N_y, step), epsrel=1e-16)[0]
        F_2[j] = result_h
    return F_2

# -------------------------- PGD approximation of U -----------------------------

def assemble_A_r(S, C, D):
    '''
       Assemblage de A_r(S) vérifiant A_r(S_n)R_n = f_r^{n-1}(S_n).
    '''
    coeff_1 = np.dot(np.dot(S, D[0]), S) * C[0]
    coeff_2 = np.dot(np.dot(S, D[3]), S) * C[3]
    coeff_3 = np.dot(np.dot(S, D[2]), S) * (C[1] + C[2])
    return coeff_1 + coeff_2 + coeff_3

def assemble_f_r(S, S_list, R_list, C, D, F_1, F_2):
    '''
       Assemblage de f_r^{n-1}(S) vérifiant A_r(S_n)R_n = f_r^{n-1}(S_n).
    '''
    result = np.dot(S, F_2) * F_1
    for k in range(1, len(S_list)):
        coeff_1 = np.dot(np.dot(S_list[k], D[0]), S) * C[0]
        coeff_2 = np.dot(np.dot(S_list[k], D[3]), S) * C[3]
        coeff_3 = np.dot(np.dot(S_list[k], D[1]), S) * C[2]
        coeff_4 = np.dot(np.dot(S_list[k], D[2]), S) * C[1]
        result -= np.dot((coeff_1 + coeff_2 + coeff_3 + coeff_4), R_list[k])
    return result

def assemble_A_s(R, C, D):
    '''
       Assemblage de A_s(R) vérifiant A_s(R_n)S_n = f_s^{n-1}(R_n).
    '''
    coeff_1 = np.dot(np.dot(R, C[0]), R) * D[0]
    coeff_2 = np.dot(np.dot(R, C[3]), R) * D[3]
    coeff_3 = np.dot(np.dot(R, C[2]), R) * (D[1] + D[2])
    return coeff_1 + coeff_2 + coeff_3

def assemble_f_s(R, S_list, R_list, C, D, F_1, F_2):
    '''
       Assemblage de f_s^{n-1}(R) vérifiant A_s(R_n)S_n = f_s^{n-1}(R_n).
    '''
    result = np.dot(R, F_1) * F_2
    for k in range(1, len(S_list)):
        coeff_1 = np.dot(np.dot(R_list[k], C[0]), R) * D[0]
        coeff_2 = np.dot(np.dot(R_list[k], C[3]), R) * D[3]
        coeff_3 = np.dot(np.dot(R_list[k], C[2]), R) * D[1]
        coeff_4 = np.dot(np.dot(R_list[k], C[1]), R) * D[2]
        result -= np.dot((coeff_1 + coeff_2 + coeff_3 + coeff_4), S_list[k])
    return result

def init_R_S(N_x, N_y, max_rand_int=max_rand_int):
    '''
       Randomly initializes the R_0 and S_0 vectors.
       Return a tuple containing two random ndarray of size N_x and N_y.
       The coefficients belong to [-max_rand_int, max_rand_int].
    '''
    return (2 * max_rand_int * np.random.rand(N_x) - max_rand_int,
            2 * max_rand_int * np.random.rand(N_y) - max_rand_int)

def H_0_norm_squared(R, S, C, E):
    '''
       Return ||R tens S||_{H_0}^2.
       Reminder : ||f||_{H_0} = ||(dx + 1/x*dy)f||_{L^2}.
    '''
    coeff_1 = np.dot(np.dot(R, C[0]), R) * np.dot(np.dot(S, E[0]), S)
    coeff_2 = np.dot(np.dot(R, C[3]), R) * np.dot(np.dot(S, E[3]), S)
    coeff_3 = 2 * np.dot(np.dot(R, C[2]), R) * np.dot(np.dot(S, E[2]), S)
    return coeff_1 + coeff_2 + coeff_3

def H_0_scalar_product(R0, S0, R1, S1, C, E):
    '''
       Return <R0 tens S0, R1 tens S1>_{H_0}.
    '''
    coeff_1 = np.dot(np.dot(R0, C[0]), R1) * np.dot(np.dot(S0, E[0]), S1)
    coeff_2 = np.dot(np.dot(R0, C[3]), R1) * np.dot(np.dot(S0, E[3]), S1)
    coeff_3 = np.dot(np.dot(R0, C[2]), R1) * np.dot(np.dot(S0, E[1]), S1)
    coeff_4 = np.dot(np.dot(R0, C[1]), R1) * np.dot(np.dot(S0, E[2]), S1)
    return coeff_1 + coeff_2 + coeff_3 + coeff_4

def H_0_diff_norm_squared(R_m0, S_m0, R_m1, S_m1, C, E):
    '''
       Return ||(R_m1 tens S_m1) - (R_m0 tens S_m0)||_{H_0}^2.
       Reminder : ||f||_{H_0} = ||(dx + 1/x*dy)f||_{L^2}.
    '''
    return (H_0_norm_squared(R_m1, S_m1, C, E) + H_0_norm_squared(R_m0, S_m0, C, E) -
            2 * H_0_scalar_product(R_m0, S_m0, R_m1, S_m1, C, E))

def fixed_point(C, D, E, F_1, F_2, R_list, S_list, N_x, N_y, eps, max_rand_int=max_rand_int):
    '''
       Return (R_n, S_n) for n >= 1.
       R is the [R_0, R_1,..., R_{n-1}] list of the consecutive iterations of the PGD algorithm.
       Likewise, S is the [S_0, S_1,..., S_{n-1}] list.
    '''
    R_m1, S_m1 = init_R_S(N_x, N_y, max_rand_int)
    R_m0 = np.zeros(N_x)
    S_m0 = np.zeros(N_y)
    counter = 0
    while(counter == 0 or (H_0_diff_norm_squared(R_m0, S_m0, R_m1, S_m1, C, E) > eps)):
        #print("Fixed point Iteration n°", counter)
        #print("Error : ", H_0_diff_norm_squared(R_m0, S_m0, R_m1, S_m1, C, E))
        for i in range(N_x):
            R_m0[i] = R_m1[i]
        for j in range(N_y):
            S_m0[j] = S_m1[j]
        A_r_S = assemble_A_r(S_m0, C, D)
        F_r_S = assemble_f_r(S_m0, S_list, R_list, C, D, F_1, F_2)
        A_r_S = sparse.csr_matrix(A_r_S)
        R_m1 = spsolve(A_r_S, F_r_S)
        A_s_R = assemble_A_s(R_m1, C, D)
        F_s_R = assemble_f_s(R_m1, S_list, R_list, C, D, F_1, F_2)
        A_s_R = sparse.csr_matrix(A_s_R)
        S_m1 = spsolve(A_s_R, F_s_R)
        counter += 1
    return(R_m1, S_m1)

def PGD(nb_iter, C, D, E, F_1, F_2, N_x, N_y, eps=1e-16, max_rand_int=max_rand_int, step = 1.):
    '''
       Return R_list and S_list such that U(x, y) =~ sum_k(sum_i(R_list[k][i]phi_i(x)) * sum_j(S_list[k][j]psi_j(y))
    '''
    R_list = [np.zeros(N_x)]
    S_list = [np.zeros(N_y)]
    for n in range(nb_iter):
        print("PGD Iteration n°", n)
        R_n, S_n = fixed_point(C, D, E, F_1, F_2, R_list, S_list, N_x, N_y, eps, max_rand_int)
        R_list += [R_n]
        S_list += [S_n]
    return (R_list, S_list)

def approximate_U_PGD(x, y, R_list, S_list, N_x, N_y, step = 1.):
    '''
       Return the approximate value of U(x, y) interpolating from (R, S).
    '''
    s = 0
    # y is 1-per.
    y = y - np.floor(y)
    for k in range(1, len(R_list)):
        r_k_x = sum([R_list[k][i] * phi(i, x, N_x, step) for i in range(N_x)])
        s_k_y = sum([S_list[k][j] * psi(j, y, N_y, step) for j in range(N_y)])
        s +=  r_k_x * s_k_y
    return s

def approximate_U_derivative_PGD(x, y, R_list, S_list, N_x, N_y, step=1.):
    '''
       Return the approximate value of dU(x, y) interpolating from (R, S).
    '''
    s = 0
    # y is 1-per.
    y = y - np.floor(y)
    for k in range(1, len(R_list)):
        r_k_x_prime = sum([R_list[k][i] * phi_prime(i, x, N_x, step) for i in range(N_x)])
        s_k_y = sum([S_list[k][j] * psi(j, y, N_y, step) for j in range(N_y)])
        r_k_x = sum([R_list[k][i] * phi(i, x, N_x, step) for i in range(N_x)])
        s_k_y_prime = sum([S_list[k][j] * psi_prime(j, y, N_y, step) for j in range(N_y)])
        s += r_k_x_prime * s_k_y + (1. / x) * r_k_x * s_k_y_prime
    return s

# -------------------------- P1 approximation of U -----------------------------

def assemble_B(C, D, N_x, N_y):
    '''
       Assemblage de B
    '''
    B = np.zeros((N_x * N_y, N_x * N_y))

    for i in range(0, N_x):
        for j in range(0, N_y):
            for k in range(0, N_x):
                for l in range(0, N_y):
                    B[K(i, j, N_y), K(k, l, N_y)] = sum([C[m][i, k] * D[m][j, l] for m in range(4)])
    return B

def assemble_U(C, D, N_x, N_y, step=1.):
    B = assemble_B(C, D, N_x, N_y)
    F = assemble_F(N_x, N_y, step)
    B = sparse.csr_matrix(B)
    return spsolve(B, F)

def approximate_U(x, y, U, N_x, N_y, step=1.):
    s=0
    # Périodicité en y
    y = y - np.floor(y)
    for i in range(0, N_x):
        for j in range(0, N_y):
            s += U[K(i, j, N_y)] * phi(i, x, N_x, step) * psi(j, y, N_y, step)
    return s

def approximate_U_derivative(x, y, U, N_x, N_y, step=1.):
    s=0
    # Périodicité en y
    y = y - np.floor(y)
    for i in range(0, N_x):
        for j in range(0, N_y):
            s += (U[K(i, j, N_y)]
                  * (phi_prime(i, x, N_x, step) * psi(j, y, N_y, step)
                     + (1.0 / x) * phi(i, x, N_x, step) * psi_prime(j, y, N_y, step)))
    return s

# ------------------------- Compute analytical solution ------------------------

inv_a = lambda x: 1.0 / a(x)
inv_aper = lambda y: 1.0 / a_per(y)
g = lambda t: inv_a(t) * scipy.integrate.quad(f, t, 1)[0]

u1 = scipy.integrate.quad(g, 0, 1, epsrel=1e-14)[0]
u1 /= -(a_per(0) * scipy.integrate.quad(inv_a, 0, 1, epsrel=1e-14)[0])  # u1 = u'(1)

def solution_analytique(x):
    integral1 = scipy.integrate.quad(g, 1, x, epsrel=1e-14)
    integral2 = scipy.integrate.quad(inv_a, 1, x, epsrel=1e-14)
    return integral1[0] + a_per(0) * u1 * integral2[0]

def deriv_analytique(x):
    return g(x) + a_per(0) * u1 * inv_a(x)

# ---------------- Show function and derivative graphs -----------------

nbPoints = 1000

X = np.linspace(0, 1, nbPoints)

C = assemble_C(N_x, step)
D = assemble_D(N_y, step)
F_1 = assemble_F_1(N_x, step)
F_2 = assemble_F_2(N_y, step)
E = assemble_E(N_y, step)
U = assemble_U(C, D, N_x, N_y, step)

R_list1, S_list1 = PGD(nb_iter, C, D, E, F_1, F_2, N_x, N_y, eps, max_rand_int, step)


Y_analytical = np.linspace(0, 1, nbPoints)
Y_analytical[1:] = [solution_analytique(x) for x in X[1:]]
Y_analytical[0] = Y_analytical[1]  # For smoother edges
Y_analytical[-1] = Y_analytical[-2]

Y_approximate = np.linspace(0, 1, nbPoints)
Y_approximate[1:] = [approximate_U(x, np.log(x), U, N_x, N_y) for x in X[1:]]
Y_approximate[0] = Y_approximate[1]
Y_approximate[-1] = Y_approximate[-2]

Y_approximate_PGD = np.linspace(0, 1, nbPoints)
Y_approximate_PGD[1:] = [approximate_U_PGD(x, np.log(x), R_list1, S_list1, N_x, N_y, step) for x in X[1:]]
Y_approximate_PGD[0] = Y_approximate_PGD[1]
Y_approximate_PGD[-1] = Y_approximate_PGD[-2]

Y_analytical_derivative = np.linspace(0, 1, nbPoints)
Y_analytical_derivative[1:] = [deriv_analytique(x) for x in X[1:]]
Y_analytical_derivative[0] = Y_analytical_derivative[1]
Y_analytical_derivative[-1] = Y_analytical_derivative[-2]

Y_approximate_derivative = np.linspace(0, 1, nbPoints)
Y_approximate_derivative[1:] = [approximate_U_derivative(x, np.log(x), U, N_x, N_y) for x in X[1:]]
Y_approximate_derivative[0] = Y_approximate_derivative[1]
Y_approximate_derivative[-1] = Y_approximate_derivative[-2]

Y_approximate_derivative_PGD = np.linspace(0, 1, nbPoints)
Y_approximate_derivative_PGD[1:] = [approximate_U_derivative_PGD(x, np.log(x), R_list1,
                                                                 S_list1, N_x, N_y, step) for x in X[1:]]
Y_approximate_derivative_PGD[0] = Y_approximate_derivative_PGD[1]
Y_approximate_derivative_PGD[-1] = Y_approximate_derivative_PGD[-2]


fig = plt.figure()
plt.plot(X, Y_analytical, color='r')
plt.plot(X, Y_approximate, color='b')
plt.plot(X, Y_approximate_PGD, color='c')
plt.xlabel('x')


fig2 = plt.figure()
plt.plot(X, Y_analytical_derivative, color='r')
plt.plot(X, Y_approximate_derivative, color='b')
plt.plot(X, Y_approximate_derivative_PGD, color='c')
plt.xlabel('x')
plt.show()


# --------------------- Error computation --------------------------

def U_P1_energy_norm_squared(U, C, D, N_x, N_y):
    '''
       Return ||U_P1||_{energy}^2.
       There is probably a cleaner way to implement this, using matrix products.
    '''
    result = 0
    for i in range(N_x):
        for j in range(N_y):
            for k in range(N_x):
                for l in range(N_y):
                    result += U[K(i, j, N_y)] * U[K(k, l, N_y)] * sum([C[m][i][k] * D[m][j][l] for m in range(4)])
    return result

def U_PGD_energy_norm_squared(R_list, S_list, C, D):
    '''
       Return ||U_PGD||_{energy}^2.
    '''
    result = 0
    for k in range(1, len(R_list)):
        for l in range(1, len(S_list)):
            result += sum([np.dot(np.dot(R_list[k], C[m]), R_list[l]) *
                           np.dot(np.dot(S_list[k], D[m]), S_list[l]) for m in range(4)])
    return result

def U_P1_U_PGD_energy_norm_scalar_product(U, R_list, S_list, C, D, N_x, N_y):
    '''
       Return <U_P1, U_PGD>_{energy}.
       Équivalence PGD : C[0] = D, C[1] = Mat(int(1/x * phi_i * phi_j')),
                         C[2] = Mat(int(1/x * phi_i' * phi_j)), C[3] = M_{1/x^2}
       Équivalence PGD : D[0] = M_aper, D[1] = Mat(int(aper * psi_i' * psi_j)),
                         D[2] = Mat(int(aper * psi_i * psi_j')), D[3] = D_aper
    '''
    result = 0
    for i in range(N_x):
        for j in range(N_y):
            for k in range(1, len(R_list)):
                result += U[K(i, j, N_y)] * sum([sum([R_list[k][l] * C[n][i][l] for l in range(N_x)]) *
                                                 sum([S_list[k][m] * D[n][j][m] for m in range(N_y)]) for n in range(4)])
    return result

def error_energy_norm(U, R_list, S_list, C, D, N_x, N_y):
    '''
       Return ||U_P1 - U_PGD||_{energy}, where U_P1 is the approximation of U using the P1 method and U_PGD the
       approximation of U using the PGD method.
       Reminder : ||f||_{energy} = ||aper*(dx + 1/x*dy)f||_{L^2}.
    '''
    return (U_P1_energy_norm_squared(U, C, D, N_x, N_y) + U_PGD_energy_norm_squared(R_list, S_list, C, D) -
            2 * U_P1_U_PGD_energy_norm_scalar_product(U, R_list, S_list, C, D, N_x, N_y)) ** (1 / 2)

# --------------------- Draw PGD error graph --------------------------
'''
C = assemble_C(N_x, step)
D = assemble_D(N_y, step)
F_1 = assemble_F_1(N_x, step)
F_2 = assemble_F_2(N_y, step)
E = assemble_E(N_y, step)
U = assemble_U(C, D, N_x, N_y, step)

N = [n for n in range(1, 42, 2)]
Y_log = []
for n in N:
    R_list, S_list = PGD(n, C, D, E, F_1, F_2, N_x, N_y, eps, max_rand_int, step)
    Y_log += [np.log(error_energy_norm(U, R_list, S_list, C, D, N_x, N_y))]

plt.plot(N, Y_log, color='r')
plt.xlabel('n')
plt.ylabel('ln(err)')
plt.show()
'''