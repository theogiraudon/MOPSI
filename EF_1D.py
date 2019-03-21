#---- importations de librairies ------
import matplotlib.pyplot as plt
import numpy as np
import time
#from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
np.set_printoptions(precision=4) # pour joli affichage des matrices

#--------------------------------
#
#     PARAMETRES DU CALCUL
#
#--------------------------------

# The set Omega is [0, R]
R = 1.

# Paramètres utilisés pour le calcul numérique de la solution analytique
N_max=500
P_max=200

def integrate(h, a, b, P):
    '''
    Integrate the function h between a and b with a method of rectangles with P points
    '''
    s = 0
    step = (b-a)/(2*P)
    for i in range(2*P):
        s += step*h(a+step/2+i*step)
    return s
    

def t_x(i, N):
    '''
    The steps of the decomposition
    '''
    return i*R/N

def a_per(x):
    return 1+np.sin(2*np.pi*x)**2
    
def a(x):
    return a_per(np.log(x))    
    
def f(x):
    return 1.
    
#---- fonctions chapeaux--------
def phi(i, y, N):
    if y<t_x(i+1, N) and y>=t_x(i, N):
        return (t_x(i+1, N)-y)/(t_x(i+1, N)-t_x(i, N))
    elif y<t_x(i, N) and y>=t_x(i-1, N):
        return (y-t_x(i-1, N))/(t_x(i, N)-t_x(i-1, N))
    else:
        return 0                        
        
def phi_prime(i, y, N):
    if y<t_x(i+1, N) and y>=t_x(i, N):
        return -1/(t_x(i+1, N)-t_x(i, N))
    elif y<t_x(i, N) and y>=t_x(i-1, N):
        return 1/(t_x(i, N)-t_x(i-1, N))
    else:
        return 0 
    
#--------------------------------------------------------------------------

def assemble_A(N, P, R=1.):
    A = lil_matrix((N-1, N-1))
    for i in range(1,N):
        for j in range(1,N):
            h = lambda y : a(y)*phi_prime(i, y, N)*phi_prime(j, y, N)
            A[i-1,j-1]=integrate(h,t_x(i-1, N),t_x(i+1, N), P)
    return A

def assemble_B(N, P, R=1.):    
    B = lil_matrix((N-1,1))        
    for i in range(1,N):
        g = lambda y : phi(i, y, N)*f(y)
        B[i-1]=integrate(g,t_x(i-1, N),t_x(i+1, N), P)
    return B

def assemble_U(N, P, R=1.):
    A = assemble_A(N, P, R)
    B = assemble_B(N, P, R)
    U = spsolve(A, B)
    return U

def approximate_solution(x, U, N, R=1.):
    '''
    Returns the approximate solution at point x
    '''
    s=0
    for i in range(N-1):
        s+=U[i]*phi(i+1, x, N)
    return s    


def approximate_derivative(x, U, N, R=1.):
    '''
    Returns the approximate solution at point x
    '''
    s=0
    for i in range(N-1):
        s+=U[i]*phi_prime(i+1, x, N)
    return s    

#--------------------------------------------------------------------------

print("Loading the table of the values of the analytic solution...")
tab_sol_anal = np.load("Val_sol_anal_1D.npy")
print("Loaded")

print("Loading the table of the values of the analytic derivative...")
tab_sol_der = np.load("Val_der_anal_1D.npy")
print("Loaded")

print("\n")

if len(tab_sol_anal) != N_max*P_max-1:
    print("The analytical solution has to be computed again")
    print("\n")

def anal_sol_interpolated(x):
    '''
    Return an approximation of the analytical solution a point x using an interpolation
    '''
    if x==1:
        return 0
    else:
        i = int(x*N_max*P_max)
        return tab_sol_anal[i]

def anal_der_interpolated(x):
    '''
    Return an approximation of the analytical derivative a point x using an interpolation
    '''
    if x==1:
        return tab_sol_der[-1]
    else:
        i = int(x*N_max*P_max)
        return tab_sol_der[i]

#--------------------------------------------------------------------------

def L2_norm(h, a, b, P_L2):
    h_squared = lambda x : h(x)**2
    return np.sqrt(integrate(h_squared, a, b, P_L2))

def L2_relative_error(U, a, b, N, R=1.):
    P_L2=50
    dif_sol_anal_et_app = lambda x : anal_sol_interpolated(x) - approximate_solution(x, U, N, R)
    return L2_norm(dif_sol_anal_et_app, a, b, P_L2)/L2_norm(anal_sol_interpolated, a, b, P_L2)    

def L2_relative_error_der(U, a, b, N, R=1.):
    P_L2=50
    dif_der_anal_et_app = lambda x : anal_der_interpolated(x) - approximate_derivative(x, U, N, R)
    return L2_norm(dif_der_anal_et_app, a, b, P_L2)/L2_norm(anal_der_interpolated, a, b, P_L2)    


#--------------------------------------------------------------------------


fig = plt.figure()

plt.xlabel("Logarithm of the step")
plt.ylabel("Logarithm of the L2 norm of the relative error")

# P is the number of points used in the function "integrate" in a interval of the form [t_x(i), t_x(i+1)]
P = 50
range_p = 4
L2_relative_error_Tab=[]
L2_relative_error_der_Tab=[]

for p in range(range_p):
    L2_relative_error_der_Tab.append([])    
    
tab_N = np.floor(np.exp(np.linspace(0,4,40)))
tab_N = np.sort(list(set(tab_N)))
tab_N = [int(N) for N in tab_N]
for N in tab_N:
    t1 = time.time()
    U = assemble_U(N, P)
    t2 = time.time()
    print("Computation time for N = {} : {} seconds".format(N, t2-t1))
    #    L2_relative_error_Tab.append(L2_relative_error(U, 0, np.exp(-p), N))
    for p in range(range_p):
        L2_relative_error_der_Tab[p].append(L2_relative_error_der(U, 0, np.exp(-p), N))
    
#N=100    
#U = assemble_U(N, P)
#X = np.linspace(0,1,1000)
#X = X[:-1]
#
#plt.plot(X, [approximate_solution(x, U, N) for x in X])    
#plt.plot(X, [anal_sol_interpolated(x) for x in X])    
#plt.show()
#plt.plot(X, [approximate_derivative(x, U, N) for x in X])    
#plt.plot(X, [anal_der_interpolated(x) for x in X])    
#plt.show()
#    
for p in range(range_p):
    print("\n")
    print("p is equal to : {}".format(p))
    plt.plot(-np.log(tab_N), np.log(L2_relative_error_der_Tab[p]))
    plt.show()
#

#--------------------------------------------------------------------------


plt.show()