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
N=7;

def t_x(i):
    return i*R/N

def a_per(x):
    return 1+np.sin(2*np.pi*x)**2
    
def a(x):
    return a_per(np.log(x))    
    
def f(x):
    return 1.    
    
def g1(x):
    return -2*x**3+3*x**2

def g2(x):
    return 2*x**3-3*x**2+1

def g3(x):
    return x**3-x**2

def g4(x):
    return x**3-2*x**2+x
       
def g1_prime(x):
    return -6*x**2+6*x

def g2_prime(x):
    return 6*x**2-6*x

def g3_prime(x):
    return 3*x**2-2*x

def g4_prime(x):
    return 3*x**2-4*x+1

def phi(i,x):
    if x<t_x(i+1) and x>=t_x(i):
        return g2((x-t_x(i))/(t_x(i+1)-t_x(i)))
    elif x<t_x(i) and x>=t_x(i-1):
        return g1((x-t_x(i-1))/(t_x(i)-t_x(i-1)))
    else:
        return 0                        

def psi(i,x):
    if x<t_x(i+1) and x>=t_x(i):
        return g4((x-t_x(i))/(t_x(i+1)-t_x(i)))*(t_x(i+1)-t_x(i))
    elif x<t_x(i) and x>=t_x(i-1):
        return g3((x-t_x(i-1))/(t_x(i)-t_x(i-1)))*(t_x(i)-t_x(i-1))
    else:
        return 0          

def phi_prime(i,x):
    if x<t_x(i+1) and x>=t_x(i):
        return g2_prime((x-t_x(i))/(t_x(i+1)-t_x(i)))/(t_x(i+1)-t_x(i))
    elif x<t_x(i) and x>=t_x(i-1):
        return g1_prime((x-t_x(i-1))/(t_x(i)-t_x(i-1)))/(t_x(i)-t_x(i-1))
    else:
        return 0                        

def psi_prime(i,x):
    if x<t_x(i+1) and x>=t_x(i):
        return g4_prime((x-t_x(i))/(t_x(i+1)-t_x(i)))
    elif x<t_x(i) and x>=t_x(i-1):
        return g3_prime((x-t_x(i-1))/(t_x(i)-t_x(i-1)))
    else:
        return 0  

#---- fonctions spline--------
#--------------------------------------------------------------------------

A = np.zeros((2*N-2,2*N-2))

for i in range(1,N):
    for j in range(1,N):
        h = lambda y : a(y)*phi_prime(i,y)*phi_prime(j,y)
        A[i-1][j-1]=scipy.integrate.quad(h,t_x(i-1),t_x(i+1))[0]

for i in range(1,N):
    for j in range(1,N):
        h = lambda y : a(y)*phi_prime(i,y)*psi_prime(j,y)
        A[i-1][N+j-2]=scipy.integrate.quad(h,t_x(i-1),t_x(i+1))[0]
        
for i in range(1,N):
    for j in range(1,N):
        h = lambda y : a(y)*psi_prime(i,y)*phi_prime(j,y)
        A[N+i-2][j-1]=scipy.integrate.quad(h,t_x(i-1),t_x(i+1))[0]
        
for i in range(1,N):
    for j in range(1,N):
        h = lambda y : a(y)*psi_prime(i,y)*psi_prime(j,y)
        A[N+i-2][N+j-2]=scipy.integrate.quad(h,t_x(i-1),t_x(i+1))[0]


B = np.zeros((2*N-2,1))
    
for i in range(1,N):
    g = lambda y : phi(i,y)*f(y)
    B[i-1]=scipy.integrate.quad(g,t_x(i-1),t_x(i+1))[0]
    
for i in range(1,N):
    g = lambda y : psi(i,y)*f(y)
    B[N+i-2]=scipy.integrate.quad(g,t_x(i-1),t_x(i+1))[0]

#-- Résolution du système linéaire --
U = np.dot(nl.inv(A),B)


def solution_approchee(y):
    s=0
    for i in range(1,N):
        s+=U[i-1]*phi(i,y)
        s+=U[N+i-2]*psi(i,y)
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