# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 14:44:31 2018

@author: giraudon
"""

#---- importations de librairies ------
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
from scipy.sparse.linalg import spsolve
from scipy import sparse
import math
np.set_printoptions(precision=4) # pour joli affichage des matrices

#--------------------------------
#
#     PARAMETRES DU CALCUL
#
#--------------------------------
R = 1.;
N_x=20;
N_y=20;


def t_x(i, N_x, R=1.):
    return i*R/(N_x+1)
    
def t_y(j, N_y, R=1.):
    return j*R/N_y
    
def a_per(y):
    return 1+np.sin(math.pi*y)**2
    
def a(x):
    if x <= 0:
        print("Invalid x value : ", x)
    return a_per(np.log(x))    
    
def f(x):
    return 1.
    
def K(i, j, N_y):
    return i*N_y + j

#---- fonctions chapeaux--------
def phi(i, x, N_x, R=1.):
    if x< t_x(i + 2, N_x, R) and x>= t_x(i + 1, N_x, R):
        return (t_x(i + 2, N_x, R) - x) / (t_x(i + 2, N_x, R) - t_x(i + 1, N_x, R))
    elif x< t_x(i + 1, N_x, R) and x>= t_x(i, N_x, R):
        return (x - t_x(i, N_x, R)) / (t_x(i + 1, N_x, R) - t_x(i, N_x, R))
    else:
        return 0                        
        
def phi_prime(i, x, N_x, R=1.):
    if x< t_x(i + 2, N_x, R) and x>= t_x(i + 1, N_x, R):
        return -1./(t_x(i + 2, N_x, R) - t_x(i + 1, N_x, R))
    elif x< t_x(i + 1, N_x, R) and x>= t_x(i, N_x, R):
        return 1./(t_x(i + 1, N_x, R) - t_x(i, N_x, R))
    else:
        return 0 

def psi(j, y, N_y, R=1.):
    if j < (N_y-1):
        if y< t_y(j + 2, N_y, R) and y>= t_y(j + 1, N_y, R):
            return (t_y(j + 2, N_y, R) - y) / (t_y(j + 2, N_y, R) - t_y(j + 1, N_y, R))
        elif y< t_y(j + 1, N_y, R) and y>= t_y(j, N_y, R):
            return (y - t_y(j, N_y, R)) / (t_y(j + 1, N_y, R) - t_y(j, N_y, R))
        else:
            return 0
    else:
        if y< t_y(1, N_y, R):
            return (t_y(1, N_y, R) - y) / (t_y(1, N_y, R) - t_y(0, N_y, R))
        elif y>= t_y(N_y - 1, N_y, R):
            return (y - t_y(N_y - 1, N_y, R)) / (t_y(N_y, N_y, R) - t_y(N_y - 1, N_y, R))
        else:
            return 0

def psi_prime(j, y, N_y, R=1.):
    if j < (N_y-1):
        if y< t_y(j + 2, N_y, R) and y>= t_y(j + 1, N_y, R):
            return -1./(t_y(j + 2, N_y, R) - t_y(j + 1, N_y, R))
        elif y< t_y(j + 1, N_y, R) and y>= t_y(j, N_y, R):
            return 1./(t_y(j + 1, N_y, R) - t_y(j, N_y, R))
        else:
            return 0
    else:
        if y< t_y(1, N_y, R):
            return -1/(t_y(1, N_y, R) - t_y(0, N_y, R))
        elif y>= t_y(N_y - 1, N_y, R):
            return 1/(t_y(i, N_y, R) - t_y(i - 1, N_y, R))
        else:
            return 0

#------------------- Solution approximation -------------------------

def assemble_C(N_x, R=1.):
    '''
       Assemblage des matrices C
    '''
    C = [np.zeros((N_x, N_x)) for k in range(4)]

    for i in range(0, N_x):
        for k in range(0, N_x):
            h = lambda x: phi_prime(i, x, N_x, R) * phi_prime(k, x, N_x, R)
            result_h = scipy.integrate.quad(h, t_x(i, N_x, R), t_x(i + 2, N_x, R), epsrel=1e-16)
            C[0][i][k] = result_h[0]

    for i in range(0, N_x):
        for k in range(0, N_x):
            h = lambda x: (1 / x) * phi(i, x, N_x, R) * phi_prime(k, x, N_x, R)
            result_h = scipy.integrate.quad(h, t_x(i, N_x, R), t_x(i + 2, N_x, R), epsrel=1e-16)
            C[1][i][k] = result_h[0]

    for i in range(0, N_x):
        for k in range(0, N_x):
            h = lambda x: (1 / x) * phi_prime(i, x, N_x, R) * phi(k, x, N_x, R)
            result_h = scipy.integrate.quad(h, t_x(i, N_x, R), t_x(i + 2, N_x, R), epsrel=1e-16)
            C[2][i][k] = result_h[0]

    for i in range(0, N_x):
        for k in range(0, N_x):
            h = lambda x: (1 / x ** 2) * phi(i, x, N_x, R) * phi(k, x, N_x, R)
            result_h = scipy.integrate.quad(h, t_x(i, N_x, R), t_x(i + 2, N_x, R), epsrel=1e-16)
            C[3][i][k] = result_h[0]
    return C

def assemble_D(N_y, R=1.):
    '''
       Assemblage des matrices D
    '''
    D = [np.zeros((N_y, N_y)) for k in range(4)]

    for j in range(0, N_y):
        for l in range(0, N_y):
            h = lambda y: a_per(y) * psi(j, y, N_y, R) * psi(l, y, N_y, R)
            if j == (N_y - 1):
                result_h = (scipy.integrate.quad(h, 0, t_y(1, N_y, R), epsrel=1e-16)[0] +
                            scipy.integrate.quad(h, t_y(N_y - 1, N_y, R), 1, epsrel=1e-16)[0])
            else:
                result_h = scipy.integrate.quad(h, t_y(j, N_y, R), t_y(j + 2, N_y, R), epsrel=1e-16)[0]
            D[0][j][l] = result_h

    for j in range(0, N_y):
        for l in range(0, N_y):
            h = lambda y: a_per(y) * psi_prime(j, y, N_y, R) * psi(l, y, N_y, R)
            if j == (N_y - 1):
                result_h = (scipy.integrate.quad(h, 0, t_y(1, N_y, R), epsrel=1e-16)[0] +
                            scipy.integrate.quad(h, t_y(N_y - 1, N_y, R), 1, epsrel=1e-16)[0])
            else:
                result_h = scipy.integrate.quad(h, t_y(j, N_y, R), t_y(j + 2, N_y, R), epsrel=1e-16)[0]
            D[1][j][l] = result_h

    for j in range(0, N_y):
        for l in range(0, N_y):
            h = lambda y: a_per(y) * psi(j, y, N_y, R) * psi_prime(l, y, N_y, R)
            if j == (N_y - 1):
                result_h = (scipy.integrate.quad(h, 0, t_y(1, N_y, R), epsrel=1e-16)[0] +
                            scipy.integrate.quad(h, t_y(N_y - 1, N_y, R), 1, epsrel=1e-16)[0])
            else:
                result_h = scipy.integrate.quad(h, t_y(j, N_y, R), t_y(j + 2, N_y, R), epsrel=1e-16)[0]
            D[2][j][l] = result_h

    for j in range(0, N_y):
        for l in range(0, N_y):
            h = lambda y: a_per(y) * psi_prime(j, y, N_y, R) * psi_prime(l, y, N_y, R)
            if j == (N_y - 1):
                result_h = (scipy.integrate.quad(h, 0, t_y(1, N_y, R), epsrel=1e-16)[0] +
                            scipy.integrate.quad(h, t_y(N_y - 1, N_y, R), 1, epsrel=1e-16)[0])
            else:
                result_h = scipy.integrate.quad(h, t_y(j, N_y, R), t_y(j + 2, N_y, R), epsrel=1e-16)[0]
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

def assemble_U(N_x, N_y, R=1.):
    C = assemble_C(N_x, R)
    D = assemble_D(N_y, R)
    B = assemble_B(C, D, N_x, N_y)
    F = assemble_F(N_x, N_y, R)
    B = sparse.csr_matrix(B)
    return spsolve(B, F)

def approximate_U(x, y, U, N_x, N_y, R=1.):
    s=0
    # Périodicité en y
    y = abs(y) - np.floor(abs(y))
    for i in range(0, N_x):
        for j in range(0, N_y):
            s += U[K(i, j, N_y)] * phi(i, x, N_x, R) * psi(j, y, N_y, R)
    return s

def approximate_U_derivative(x, y, U, N_x, N_y, R=1.):
    s=0
    # Périodicité en y
    y = abs(y) - np.floor(abs(y))
    for i in range(0, N_x):
        for j in range(0, N_y):
            s += (U[K(i, j, N_y)]
                  * (phi_prime(i, x, N_x, R) * psi(j, y, N_y, R)
                     + (1.0 / x) * phi(i, x, N_x, R) * psi_prime(j, y, N_y, R)))
    return s


#--------------------------------------------------------------------------
  
    
fig = plt.figure()

nbPoints = 1000

X = np.linspace(0, 1, nbPoints)

#---Solution analytique------------------------

#inv_a = lambda x : 1.0/a(x)
#g = lambda t : inv_a(t)*(scipy.integrate.quad(f,1,t, epsrel = 1e-14)[0])
#u1 = scipy.integrate.quad(g,0,1, epsrel = 1e-14)[0]
#u1 /= -scipy.integrate.quad(inv_a,0,1, epsrel = 1e-14)[0]


inv_a = lambda x : 1.0/a(x)
inv_aper = lambda y : 1.0/a_per(y)
g = lambda t : inv_a(t)*t

g1 = lambda t : np.exp(2*t)/a_per(t)
g2 = lambda t : np.exp(t)/a_per(t)

N = 400000
xvec = np.linspace(-40,0,N)
g1vec = g1(xvec)
g2vec = g2(xvec)

u3 = scipy.integrate.trapz(g1vec,xvec)
u3 /= scipy.integrate.trapz(g2vec,xvec)

print("u3 = ", u3)

u2 = scipy.integrate.quad(g1,-40,0, epsrel = 1e-14)[0]  
u2 /= scipy.integrate.quad(g2,-40,0, epsrel = 1e-14)[0]  

print("u2 = ", u2)

u1 = scipy.integrate.quad(g,0,1, epsrel = 1e-14)[0]
u1 /= -scipy.integrate.quad(inv_a,0,1, epsrel = 1e-14)[0]

print("u1 = ", u1)


def solution_analytique(x):
    integral1 = scipy.integrate.quad(g,1,x, epsrel = 1e-14)
    integral2 = scipy.integrate.quad(inv_a,1,x, epsrel = 1e-14)
    return integral1[0] + u1*integral2[0]

def deriv_analytique(x):
    return (x-u3)/a(x)

#Y_analytique = [-solution_analytique(x) for x in X]
Y_analytique = [-deriv_analytique(x) for x in X]

plt.plot(X,Y_analytique,color='r')
#plt.show()

'''
fig = plt.figure()

Uvec = np.zeros((N_x+2,N_y+1))
for i in range(0,N_x+2):
    for j in range(0,N_y+1):
        Uvec[i,j] = solution_approchee(t_x(i, N_x, R), t_y(j, N_y, R))
'''
#plt.imshow(Uvec)
#plt.show()


#---------------- Show function and derivative graphs -----------------
'''
U = assemble_U(N_x, N_y, R)
Y_2 = np.linspace(0,1,nbPoints)
Y_2[0] = 0
#Y[1:] = [solution_approchee(x, np.log(x)) for x in X[1:]]
Y_2[1:] = [approximate_U_derivative(x, np.log(x), U, N_x, N_y) for x in X[1:]]


fig = plt.figure()
plt.plot(X,Y_analytique,color='r')
plt.xlabel('x')
plt.plot(X, Y_2, color='g')'''

plt.show()


#--------------------- L^2 Error computation --------------------------

def L_2_norm(f, a=0, b=1, epsrel=1.49e-08):
    '''
       Return the approximate L^2 norm of f restricted to (inf, sup) using
       nb_points points in the range.
    '''
    f_squared = lambda x : f(x)**2    # f(x) is real
    return scipy.integrate.quad(f_squared, a, b, epsrel=epsrel)[0]**(1/2)


