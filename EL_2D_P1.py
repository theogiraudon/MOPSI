# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 14:44:31 2018

@author: giraudon
"""

#---- importations de librairies ------
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as nl
import scipy
np.set_printoptions(precision=4) # pour joli affichage des matrices

#--------------------------------
#
#     PARAMETRES DU CALCUL
#
#--------------------------------
R = 1.;
N_x=130;
N_y=130;

def t_x(i):
    return i*R/(N_x+1)
    
def t_y(j):
    return j*R/N_y
    
def a_per(y):
    return 1+np.sin(y)**2
    
def a(x):
    return a_per(np.log(x))    
    
def f(x):
    return 1.
    
def K(i,j):
    return i*N_y + j
    
#---- fonctions chapeaux--------
def phi(i,x):
    if x<t_x(i+1) and x>=t_x(i):
        return (t_x(i+1)-x)/(t_x(i+1)-t_x(i))
    elif x<t_x(i) and x>=t_x(i-1):
        return (x-t_x(i-1))/(t_x(i)-t_x(i-1))
    else:
        return 0                        
        
def phi_prime(i,x):
    if x<t_x(i+1) and x>=t_x(i):
        return -1/(t_x(i+1)-t_x(i))
    elif x<t_x(i) and x>=t_x(i-1):
        return 1/(t_x(i)-t_x(i-1))
    else:
        return 0 

def psi(j,y):
    if j < N_y:
        if y<t_y(j+1) and y>=t_y(j):
            return (t_y(j+1)-y)/(t_y(j+1)-t_y(j))
        elif y<t_y(j) and y>=t_y(j-1):
            return (y-t_y(j-1))/(t_y(j)-t_y(j-1))
        else:
            return 0
    else:
        if y<t_y(1):
            return (t_y(1)-y)/(t_y(1)-t_y(0))
        elif y>=t_y(N_y-1):
            return (y-t_y(j-1))/(t_y(j)-t_y(j-1))
        else:
            return 0

def psi_prime(j,y):
    if j < N_y:
        if y<t_y(j+1) and y>=t_y(j):
            return -1/(t_y(j+1)-t_y(j))
        elif y<t_y(j) and y>=t_y(j-1):
            return 1/(t_y(j)-t_y(j-1))
        else:
            return 0
    else:
        if y<t_y(1):
            return -1/(t_y(1)-t_y(0))
        elif y>=t_y(N_y-1):
            return 1/(t_y(i)-t_y(i-1))
        else:
            return 0 
#--------------------------------------------------------------------------

B = np.zeros((N_x*N_y,N_x*N_y))
C = [np.zeros((N_x, N_x)) for k in range(4)]
D = [np.zeros((N_y, N_y)) for k in range(4)] 
F = np.zeros(N_x*N_y)

#---Assemblage des matrices C
for i in range(1,N_x+1):
    for k in range(1,N_x+1):
        h = lambda x : phi_prime(i,x)*phi_prime(k,x)
        result_h=scipy.integrate.quad(h,t_x(i-1),t_x(i+1))
        C[0][i-1][k-1]=result_h[0]
        
for i in range(1,N_x+1):
    for k in range(1,N_x+1):
        h = lambda x : (1/x)*phi(i,x)*phi_prime(k,x)
        result_h=scipy.integrate.quad(h,t_x(i-1),t_x(i+1))
        C[1][i-1][k-1]=result_h[0]

for i in range(1,N_x+1):
    for k in range(1,N_x+1):
        h = lambda x : (1/x)*phi_prime(i,x)*phi(k,x)
        result_h=scipy.integrate.quad(h,t_x(i-1),t_x(i+1))
        C[2][i-1][k-1]=result_h[0]

for i in range(1,N_x+1):
    for k in range(1,N_x+1):
        h = lambda x : (1/x**2)*phi(i,x)*phi(k,x)
        result_h=scipy.integrate.quad(h,t_x(i-1),t_x(i+1))
        C[3][i-1][k-1]=result_h[0]        

#---Assemblage des matrices D
for j in range(1,N_y+1):
    for l in range(1,N_y+1):
        h = lambda y : a_per(y)*psi(j,y)*psi(l,y)
        if j == N_y:
            result_h=(scipy.integrate.quad(h,0,t_y(1))[0]+
                     scipy.integrate.quad(h,t_y(N_y-1),1)[0])
        else:
            result_h=scipy.integrate.quad(h,t_y(j-1),t_y(j+1))[0]
        D[0][j-1][l-1]=result_h

for j in range(1,N_y+1):
    for l in range(1,N_y+1):
        h = lambda y : a_per(y)*psi_prime(j,y)*psi(l,y)
        if j == N_y:
            result_h=(scipy.integrate.quad(h,0,t_y(1))[0]+
                     scipy.integrate.quad(h,t_y(N_y-1),1)[0])
        else:
            result_h=scipy.integrate.quad(h,t_y(j-1),t_y(j+1))[0]
        D[1][j-1][l-1]=result_h
        
for j in range(1,N_y+1):
    for l in range(1,N_y+1):
        h = lambda y : a_per(y)*psi(j,y)*psi_prime(l,y)
        if j == N_y:
            result_h=(scipy.integrate.quad(h,0,t_y(1))[0]+
                     scipy.integrate.quad(h,t_y(N_y-1),1)[0])
        else:
            result_h=scipy.integrate.quad(h,t_y(j-1),t_y(j+1))[0]
        D[2][j-1][l-1]=result_h
        
for j in range(1,N_y+1):
    for l in range(1,N_y+1):
        h = lambda y : a_per(y)*psi_prime(j,y)*psi_prime(l,y)
        if j == N_y:
            result_h=(scipy.integrate.quad(h,0,t_y(1))[0]+
                     scipy.integrate.quad(h,t_y(N_y-1),1)[0])
        else:
            result_h=scipy.integrate.quad(h,t_y(j-1),t_y(j+1))[0]
        D[3][j-1][l-1]=result_h

#---Assemblage de F
for i in range(1,N_x+1):
    for j in range(1,N_y+1):
        g = lambda x : phi(i,x)*f(x)
        h = lambda y : psi(j,y)
        
        result_g=scipy.integrate.quad(g,t_x(i-1),t_x(i+1))[0]
        if j == N_y:
            result_h=(scipy.integrate.quad(h,0,t_y(1))[0]+
                     scipy.integrate.quad(h,t_y(N_y-1),1)[0])
        else:
            result_h=scipy.integrate.quad(h,t_y(j-1),t_y(j+1))[0]       
        F[K(i-1,j-1)]=result_g*result_h
        
        
#---Assemblage de B
for i in range(N_x):
    for j in range(N_y):
        for k in range(N_x):
            for l in range(N_y):
                B[K(i,j),K(k,l)] = sum([C[m][i,k]*D[m][j,l] for m in range(4)])


#-- Résolution du système linéaire --
U = np.dot(nl.inv(B),F)

def solution_approchee(x,y):
    s=0
    # Périodicité en y
    y = abs(y) - np.floor(abs(y))
    for i in range(1,N_x+1):
        for j in range(1,N_y+1):
            s+=U[K(i-1,j-1)]*phi(i,x)*psi(j,y)
    return s    
    
#--------------------------------------------------------------------------
  
    
fig = plt.figure()

nbPoints = 1000

X = np.linspace(0,1,nbPoints)
Y = np.linspace(0,1,nbPoints)
Y[0] = 0
Y[1:] = [solution_approchee(x, np.log(x)) for x in X[1:]]

plt.xlabel('x')
plt.plot(X,Y,color='b')
#---Solution analytique------------------------

inv_a = lambda x : 1/a(x)
g = lambda t : inv_a(t)*(scipy.integrate.quad(f,1,t)[0])
u1 = scipy.integrate.quad(g,0,1)[0]
u1 /= -scipy.integrate.quad(inv_a,0,1)[0]

def solution_analytique(x):
    integral1 = scipy.integrate.quad(g,1,x)
    integral2 = scipy.integrate.quad(inv_a,1,x)
    return integral1[0] + u1*integral2[0]
    
Y_analytique = [-solution_analytique(x) for x in X]
plt.plot(X,Y_analytique,color='r')