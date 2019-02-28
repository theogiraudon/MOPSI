#---- importations de librairies ------
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as nl
np.set_printoptions(precision=4) # pour joli affichage des matrices

import sol_anal_1D

#--------------------------------
#
#     PARAMETRES DU CALCUL
#
#--------------------------------
R = 1.
N=10
P=30

def integrate(h, a, b):
    s = 0
    step = (b-a)/(2*P)
    for i in range(2*P):
        s += step*h(a+step/2+i*step)
    return s
    
def x(i):
    return i*R/N

def a_per(x):
    return 1+np.sin(2*np.pi*x)**2
    
def a(x):
    return a_per(np.log(x))    
    
def f(x):
    return 1.
    
#---- fonctions chapeaux--------
def phi(i,y):
    if y<x(i+1) and y>=x(i):
        return (x(i+1)-y)/(x(i+1)-x(i))
    elif y<x(i) and y>=x(i-1):
        return (y-x(i-1))/(x(i)-x(i-1))
    else:
        return 0                        
        
def phi_prime(i,y):
    if y<x(i+1) and y>=x(i):
        return -1/(x(i+1)-x(i))
    elif y<x(i) and y>=x(i-1):
        return 1/(x(i)-x(i-1))
    else:
        return 0 
    
#--------------------------------------------------------------------------

A = np.zeros((N-1,N-1))

for i in range(1,N):
    for j in range(1,N):
        h = lambda y : a(y)*phi_prime(i,y)*phi_prime(j,y)
        A[i-1][j-1]=integrate(h,x(i-1),x(i+1))


B = np.zeros((N-1,1))
    
for i in range(1,N):
    g = lambda y : phi(i,y)*f(y)
    B[i-1]=integrate(g,x(i-1),x(i+1))

#-- Résolution du système linéaire --
U = np.dot(nl.inv(A),B)

def solution_approchee(y):
    s=0
    for i in range(N-1):
        s+=U[i]*phi(i+1,y)
    return s    

#--------------------------------------------------------------------------

fig = plt.figure()

X = np.linspace(0,1,1000)
Y = []

for elt in X:
    Y.append(solution_approchee(elt))    

plt.xlabel('x')

plt.plot(X,Y)
    
Y_analytique = [sol_anal_1D.solution_analytique(x) for x in X]
plt.plot(X,Y_analytique,color='r')
plt.show()
