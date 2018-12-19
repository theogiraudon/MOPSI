#---- importations de librairies ------
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as nl
import scipy.integrate
np.set_printoptions(precision=4) # pour joli affichage des matrices

#--------------------------------
#
#     PARAMETRES DU CALCUL
#
#--------------------------------
R = 1.;
N=10;

def x(i):
    return i*R/N

def a_per(x):
    return 1+np.sin(np.pi*x)**2
    
def a(x):
    return a_per(np.log(x))    
    
def f(x):
    return 1.    
    
    
def coef_spline_phi(i, side):
    if side=='l':        
        M = [[x(i)**3,x(i)**2,x(i),1],
             [3*x(i)**2,2*x(i),1,0],
              [x(i-1)**3,x(i-1)**2,x(i-1),1],
               [3*x(i-1)**2,2*x(i-1),1,0]]
        B = [1,0,0,0]
    elif side=='r':
        M = [[x(i+1)**3,x(i+1)**2,x(i+1),1],
             [3*x(i+1)**2,2*x(i+1),1,0],
              [x(i)**3,x(i)**2,x(i),1],
               [3*x(i)**2,2*x(i),1,0]]
        B = [0,0,1,0]
    return nl.solve(M,B)
    
def coef_spline_psi(i, side):
    if side=='l':
        M = [[x(i)**3,x(i)**2,x(i),1],
                 [3*x(i)**2,2*x(i),1,0],
                  [x(i-1)**3,x(i-1)**2,x(i-1),1],
                   [3*x(i-1)**2,2*x(i-1),1,0]]
        B = [0,0,0,1]
    elif side=='r':
        M = [[x(i+1)**3,x(i+1)**2,x(i+1),1],
             [3*x(i+1)**2,2*x(i+1),1,0],
              [x(i)**3,x(i)**2,x(i),1],
               [3*x(i)**2,2*x(i),1,0]]
        B = [0,1,0,0]
    return nl.solve(M,B)
       
#---- fonctions spline--------
def phi(i,y):
    if y<x(i+1) and y>=x(i):
        coef = coef_spline_phi(i,'r')
        return y**3*coef[0]+y**2*coef[1]+y*coef[2]+coef[3]
    elif y<x(i) and y>=x(i-1):
        coef = coef_spline_phi(i,'l')
        return y**3*coef[0]+y**2*coef[1]+y*coef[2]+coef[3]
    else:
        return 0                        
        
def phi_prime(i,y):
    if y<x(i+1) and y>=x(i):
        coef = coef_spline_phi(i,'r')
        return 2*y**2*coef[0]+y*coef[1]+coef[2]
    elif y<x(i) and y>=x(i-1):
        coef = coef_spline_phi(i,'l')
        return 2*y**2*coef[0]+y*coef[1]+coef[2]
    else:
        return 0 
        
def psi(i,y):
    if y<x(i+1) and y>=x(i):
        coef = coef_spline_psi(i,'r')
        return y**3*coef[0]+y**2*coef[1]+y*coef[2]+coef[3]
    elif y<x(i) and y>=x(i-1):
        coef = coef_spline_psi(i,'l')
        return y**3*coef[0]+y**2*coef[1]+y*coef[2]+coef[3]
    else:
        return 0
        
def psi_prime(i,y):
    if y<x(i+1) and y>=x(i):
        coef = coef_spline_psi(i,'r')
        return 2*y**2*coef[0]+y*coef[1]+coef[2]
    elif y<x(i) and y>=x(i-1):
        coef = coef_spline_psi(i,'l')
        return 2*y**2*coef[0]+y*coef[1]+coef[2]
    else:
        return 0
    
#--------------------------------------------------------------------------

A = np.zeros((2*N,2*N))

for i in range(1,N+1):
    for j in range(1,N+1):
        h = lambda y : a(y)*phi_prime(i,y)*phi_prime(j,y)
        A[i-1][j-1]=scipy.integrate.quad(h,x(i-1),x(i+1))[0]

for i in range(1,N+1):
    for j in range(1,N+1):
        h = lambda y : a(y)*phi_prime(i,y)*psi_prime(j,y)
        A[i-1][N+j-1]=scipy.integrate.quad(h,x(i-1),x(i+1))[0]
        
for i in range(1,N+1):
    for j in range(1,N+1):
        h = lambda y : a(y)*psi_prime(i,y)*phi_prime(j,y)
        A[N+i-1][j-1]=scipy.integrate.quad(h,x(i-1),x(i+1))[0]
        
for i in range(1,N+1):
    for j in range(1,N+1):
        h = lambda y : a(y)*psi_prime(i,y)*psi_prime(j,y)
        A[N+i-1][N+j-1]=scipy.integrate.quad(h,x(i-1),x(i+1))[0]


B = np.zeros((2*N,1))
    
for i in range(1,N+1):
    g = lambda y : phi(i,y)*f(y)
    B[i-1]=scipy.integrate.quad(g,x(i-1),x(i+1))[0]
    
for i in range(1,N+1):
    g = lambda y : psi(i,y)*f(y)
    B[N+i-1]=scipy.integrate.quad(g,x(i-1),x(i+1))[0]

#-- Résolution du système linéaire --
U = np.dot(nl.inv(A),B)

def solution_approchee(y):
    s=0
    for i in range(1,N+1):
        s+=U[i-1]*phi(i,y)
        s+=U[N+i-1]*psi(i,y)
    return s    

#--------------------------------------------------------------------------

fig = plt.figure()

X = np.linspace(0,1,10000)
Y = []

for elt in X:
    Y.append(solution_approchee(elt))
    

plt.xlabel('x')

plt.plot(X,Y)

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
plt.show()