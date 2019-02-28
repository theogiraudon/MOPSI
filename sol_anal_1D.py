#---- importations de librairies ------
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=4) # pour joli affichage des matrices

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

inv_a = lambda x : 1/a(x)
g = lambda t : -inv_a(t)*(integrate(f,t,1))
u1 = integrate(g,0,1)
u1 /= -integrate(inv_a,0,1)

def solution_analytique(x):
    return integrate(g,x,1) + u1*integrate(inv_a,x,1)
    
X = np.linspace(0,1,1000)
plt.plot(X,solution_analytique(X))