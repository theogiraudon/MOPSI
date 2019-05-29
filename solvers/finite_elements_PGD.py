# -*- coding: utf-8 -*-
"""
Created on Tue Feb 5 13:40:19 2019

@author: Lauret Jérémy
"""

# ---- importations de librairies ------
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve
from time import time

from core.finite_elements import tridiag, tridiag2
from core.parameters import a, a_per, f, P, N_x, N_y, nb_iter, eps, max_rand_int, N_max, P_max
from core.finite_elements import t_x, t_y, phi, phi2D, phi2D_prime, phi_prime, psi2D, psi2D_prime, K
from core.integrate import rectangle_midpoints
from solvers.finite_elements_2d import assemble_C, assemble_D, assemble_B, assemble_F, assemble_U_2D, approximate_solution_2D, approximate_derivative_2D


np.set_printoptions(precision=4)  # pour joli affichage des matrices

def assemble_F_1(N_x, P):
    '''
       Assemblage de la matrice F_1.
    '''
    g = lambda x: phi(1, x, N_x + 1)

    int_phi = rectangle_midpoints(g, t_x(0, N_x + 1), t_x(2, N_x + 1), N_x + 1, P)

    return int_phi*np.ones((N_x))

def assemble_F_2(N_y, P):
    '''
       Assemblage de la matrice F_2.
    '''
    h = lambda y: psi2D(0, y, N_y)

    int_psi = rectangle_midpoints(h, t_x(0, N_x + 1), t_x(2, N_x + 1), N_x + 1, P)

    return int_psi*np.ones((N_y))


# -------------------------- PGD approximation of U -----------------------------

def assemble_A_r(S, C, D):
    '''
       Assemblage de A_r(S) vérifiant A_r(S_n)R_n = f_r^{n-1}(S_n).
    '''
    coeff_1 = (D[0]@S)@S * C[0]
    coeff_2 = (D[3]@S)@S * C[3]
    coeff_3 = (D[2]@S)@S * (C[1] + C[2])
    return coeff_1 + coeff_2 + coeff_3

def assemble_f_r(S, S_list, R_list, C, D, F_1, F_2):
    '''
       Assemblage de f_r^{n-1}(S) vérifiant A_r(S_n)R_n = f_r^{n-1}(S_n).
    '''
    result = np.dot(S, F_2) * F_1
    for k in range(1, len(S_list)):
        coeff_1 = (D[0]@S_list[k])@S * C[0]
        coeff_2 = (D[3]@S_list[k])@S * C[3]
        coeff_3 = (D[1]@S_list[k])@S * C[2]
        coeff_4 = (D[2]@S_list[k])@S * C[1]
        result -= R_list[k]@(coeff_1 + coeff_2 + coeff_3 + coeff_4)
    return result

def assemble_A_s(R, C, D):
    '''
       Assemblage de A_s(R) vérifiant A_s(R_n)S_n = f_s^{n-1}(R_n).
    '''
    coeff_1 = (C[0]@R)@R * D[0]
    coeff_2 = (C[3]@R)@R * D[3]
    coeff_3 = (C[2]@R)@R * (D[1] + D[2])
    return coeff_1 + coeff_2 + coeff_3

def assemble_f_s(R, S_list, R_list, C, D, F_1, F_2):
    '''
       Assemblage de f_s^{n-1}(R) vérifiant A_s(R_n)S_n = f_s^{n-1}(R_n).
    '''
    result = np.dot(R, F_1) * F_2
    for k in range(1, len(S_list)):
        coeff_1 = (C[0]@R_list[k])@R * D[0]
        coeff_2 = (C[3]@R_list[k])@R * D[3]
        coeff_3 = (C[2]@R_list[k])@R * D[1]
        coeff_4 = (C[1]@R_list[k])@R * D[2]
        result -= S_list[k]@(coeff_1 + coeff_2 + coeff_3 + coeff_4)
    return result

def init_R_S(N_x, N_y, max_rand_int=max_rand_int):
    '''
       Randomly initializes the R_0 and S_0 vectors.
       Return a tuple containing two random ndarray of size N_x and N_y.
       The coefficients belong to [-max_rand_int, max_rand_int].
    '''
    return (2 * max_rand_int * np.random.rand(N_x) - max_rand_int,
            2 * max_rand_int * np.random.rand(N_y) - max_rand_int)

def H_0_norm_squared(R, S, C, D):
    '''
       Return ||R tens S||_{H_0}^2.
       Reminder : ||f||_{H_0} = ||(dx + 1/x*dy)f||_{L^2}.
    '''
    coeff_1 = (C[0]@R)@R * (D[0]@S)@S
    coeff_2 = (C[3]@R)@R * (D[3]@S)@S
    coeff_3 = 2 * (C[2]@R)@R * (D[2]@S)@S
    return coeff_1 + coeff_2 + coeff_3

def H_0_scalar_product(R0, S0, R1, S1, C, D):
    '''
       Return <R0 tens S0, R1 tens S1>_{H_0}.
    '''
    coeff_1 = (C[0]@R0)@R1 * (D[0]@S0)@S1
    coeff_2 = (C[3]@R0)@R1 * (D[3]@S0)@S1
    coeff_3 = (C[2]@R0)@R1 * (D[1]@S0)@S1
    coeff_4 = (C[1]@R0)@R1 * (D[2]@S0)@S1
    return coeff_1 + coeff_2 + coeff_3 + coeff_4

def H_0_diff_norm_squared(R_m0, S_m0, R_m1, S_m1, C, D):
    '''
       Return ||(R_m1 tens S_m1) - (R_m0 tens S_m0)||_{H_0}^2.
       Reminder : ||f||_{H_0} = ||(dx + 1/x*dy)f||_{L^2}.
    '''
    return (H_0_norm_squared(R_m1, S_m1, C, D) + H_0_norm_squared(R_m0, S_m0, C, D) -
            2 * H_0_scalar_product(R_m0, S_m0, R_m1, S_m1, C, D))

def fixed_point(C, D, F_1, F_2, R_list, S_list, N_x, N_y, eps, max_rand_int=max_rand_int):
    '''
       Return (R_n, S_n) for n >= 1.
       R is the [R_0, R_1,..., R_{n-1}] list of the consecutive iterations of the PGD algorithm.
       Likewise, S is the [S_0, S_1,..., S_{n-1}] list.
    '''
    R_m1, S_m1 = init_R_S(N_x, N_y, max_rand_int)
    R_m0 = np.zeros(N_x)
    S_m0 = np.zeros(N_y)
    counter = 0
    while(counter == 0 or (H_0_diff_norm_squared(R_m0, S_m0, R_m1, S_m1, C, D) > eps)):
        #print("Fixed point Iteration n°", counter)
        #print("Error : ", H_0_diff_norm_squared(R_m0, S_m0, R_m1, S_m1, C, E))
        for i in range(N_x):
            R_m0[i] = R_m1[i]
        for j in range(N_y):
            S_m0[j] = S_m1[j]
        A_r_S = assemble_A_r(S_m0, C, D)
        F_r_S = assemble_f_r(S_m0, S_list, R_list, C, D, F_1, F_2)
        A_r_S = csc_matrix(A_r_S)
        R_m1 = spsolve(A_r_S, F_r_S)
        A_s_R = assemble_A_s(R_m1, C, D)
        F_s_R = assemble_f_s(R_m1, S_list, R_list, C, D, F_1, F_2)
        A_s_R = csc_matrix(A_s_R)
        S_m1 = spsolve(A_s_R, F_s_R)
        counter += 1
    return(R_m1, S_m1)

def PGD(N_x, N_y, nb_iter=nb_iter, max_rand_int=max_rand_int):
    '''
       Return R_list and S_list such that U(x, y) =~ sum_k(sum_i(R_list[k][i]phi_i(x)) * sum_j(S_list[k][j]psi_j(y))
    '''
    C = assemble_C(N_x, P)
    D = assemble_D(N_y, P)
    F_1 = assemble_F_1(N_x, P)
    F_2 = assemble_F_2(N_y, P)
    R_list = [np.zeros(N_x)]
    S_list = [np.zeros(N_y)]
    for n in range(nb_iter):
        print("PGD Iteration n°", n)
        R_n, S_n = fixed_point(C, D, F_1, F_2, R_list, S_list, N_x, N_y, eps, max_rand_int)
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
        r_k_x = sum([R_list[k][i - 1] * phi(i, x, N_x + 1) for i in range(1, N_x + 1)])
        s_k_y = sum([S_list[k][j] * psi2D(j, y, N_y) for j in range(N_y)])
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
        r_k_x_prime = sum([R_list[k][i - 1] * phi_prime(i, x, N_x + 1) for i in range(1, N_x + 1)])
        s_k_y = sum([S_list[k][j] * psi2D(j, y, N_y) for j in range(N_y)])
        r_k_x = sum([R_list[k][i - 1] * phi(i, x, N_x + 1) for i in range(1, N_x + 1)])
        s_k_y_prime = sum([S_list[k][j] * psi2D_prime(j, y, N_y) for j in range(N_y)])
        s += r_k_x_prime * s_k_y + (1. / x) * r_k_x * s_k_y_prime
    return s


# --------------------- Error computation --------------------------

def coef(n, U, C, D, i, j, k, l):
    return U[K(i, j, N_y)]*U[K(k, l, N_y)]*C[n][i,k]*D[n][j,l]



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
                    result += U[K(i, j, N_y)] * U[K(k, l, N_y)] * sum([C[m][i,k] * D[m][j,l] for m in range(4)])
    return result

def U_PGD_energy_norm_squared(R_list, S_list, C, D):
    '''
       Return ||U_PGD||_{energy}^2.
    '''
    result = 0
    for k in range(1, len(R_list)):
        for l in range(1, len(S_list)):
            result += sum([(C[m]@R_list[k])@R_list[l] * (D[m]@S_list[k])@S_list[l] for m in range(4)])
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
                result += U[K(i, j, N_y)] * sum([sum([R_list[k][l] * C[n][i,l] for l in range(N_x)]) *
                                                 sum([S_list[k][m] * D[n][j,m] for m in range(N_y)]) for n in range(4)])
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

def display_errors_PGD(start, stop, step):
    """
    :param start: first n (n is the number of iterations in the PGD method)
    :param stop: last n
    :param step: step between the different values of n
    """
    n_list = range(start, stop, step)
    t0 = time()
    print("Computing U...")
    C = assemble_C(N_x, P)
    D = assemble_D(N_y, P)
    U = assemble_U_2D(N_x, N_y, P)
    print("U computation time (N_x = {}, N_y = {}, P = {}) : {} seconds".format(N_x, N_y, P, time() - t0), flush=True)

    Y_log = []
    for n in n_list:
        t0 = time()
        R_list, S_list = PGD(N_x, N_y, nb_iter=n)
        print("R and S computation time (n = {}) : {} seconds".format(n, time() - t0), flush=True)
        t0 = time()
        Y_log.append(np.log(error_energy_norm(U, R_list, S_list, C, D, N_x, N_y)))
        print('Time to do the error: ', time() - t0)

    plt.plot(np.log(n_list), Y_log, color='r')
    plt.xlabel('log of the number of iterations')
    plt.ylabel('log of the energy error')
    plt.show()




