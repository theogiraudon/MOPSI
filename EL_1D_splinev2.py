#---- importations de librairies ------
import matplotlib.pyplot as plt
import numpy as np
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
N_max=15
P_max=100

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

def phi(i, x, N):
    if x<t_x(i+1, N) and x>=t_x(i, N):
        return g2((x-t_x(i, N))/(t_x(i+1, N)-t_x(i, N)))
    elif x<t_x(i, N) and x>=t_x(i-1, N):
        return g1((x-t_x(i-1, N))/(t_x(i, N)-t_x(i-1, N)))
    else:
        return 0                        

def psi(i, x, N):
    if x<t_x(i+1, N) and x>=t_x(i, N):
        return g4((x-t_x(i, N))/(t_x(i+1, N)-t_x(i, N)))*(t_x(i+1, N)-t_x(i, N))
    elif x<t_x(i, N) and x>=t_x(i-1, N):
        return g3((x-t_x(i-1, N))/(t_x(i, N)-t_x(i-1, N)))*(t_x(i, N)-t_x(i-1, N))
    else:
        return 0          

def phi_prime(i, x, N):
    if x<t_x(i+1, N) and x>=t_x(i, N):
        return g2_prime((x-t_x(i, N))/(t_x(i+1, N)-t_x(i, N)))/(t_x(i+1, N)-t_x(i, N))
    elif x<t_x(i, N) and x>=t_x(i-1, N):
        return g1_prime((x-t_x(i-1, N))/(t_x(i, N)-t_x(i-1, N)))/(t_x(i, N)-t_x(i-1, N))
    else:
        return 0                        

def psi_prime(i, x, N):
    if x<t_x(i+1, N) and x>=t_x(i, N):
        return g4_prime((x-t_x(i, N))/(t_x(i+1, N)-t_x(i, N)))
    elif x<t_x(i, N) and x>=t_x(i-1, N):
        return g3_prime((x-t_x(i-1, N))/(t_x(i, N)-t_x(i-1, N)))
    else:
        return 0  

    
#--------------------------------------------------------------------------

def assemble_A(N, P, R=1.):
    A = lil_matrix((2*N-2, 2*N-2))    
    for i in range(1,N):
        for j in range(1,N):
            h = lambda x : a(x)*phi_prime(i, x, N)*phi_prime(j, x, N)
            A[i-1, j-1]=integrate(h,t_x(i-1, N),t_x(i+1, N), P)
    
    for i in range(1,N):
        for j in range(1,N):
            h = lambda x : a(x)*phi_prime(i, x, N)*psi_prime(j, x, N)
            A[i-1, N+j-2]=integrate(h,t_x(i-1, N),t_x(i+1, N), P)
            
    for i in range(1,N):
        for j in range(1,N):
            h = lambda x : a(x)*psi_prime(i, x, N)*phi_prime(j, x, N)
            A[N+i-2, j-1]=integrate(h,t_x(i-1, N),t_x(i+1, N), P)
            
    for i in range(1,N):
        for j in range(1,N):
            h = lambda x : a(x)*psi_prime(i, x, N)*psi_prime(j, x, N)
            A[N+i-2, N+j-2]=integrate(h,t_x(i-1, N),t_x(i+1, N), P)
            
    return A

def assemble_B(N, P, R=1.):    
    B = lil_matrix((2*N-2,1))        
    for i in range(1,N):
        g = lambda x : phi(i, x, N)*f(x)
        B[i-1]=integrate(g,t_x(i-1, N),t_x(i+1, N), P)
    for i in range(1,N):
        g = lambda x : psi(i, x, N)*f(x)
        B[N+i-2]=integrate(g,t_x(i-1, N),t_x(i+1, N), P)
    return B

def assemble_U(N, P, R=1.):
    A = assemble_A(N, P, R)
    B = assemble_B(N, P, R)
    U = spsolve(A, B)
    return U

def approximate_solution(x, U, N, P, R=1.):
    '''
    Returns the approximate solution at point x
    '''
    s=0
    for i in range(1,N):
        s+=U[i-1]*phi(i, x, N)
        s+=U[N+i-2]*psi(i, x, N)
    return s    
    

#--------------------------------------------------------------------------
#
#file = open("Val_sol_anal.txt","r")
#
#def recuperation_tableau_sol_anal():
#    tab_str=[]
#    for c in file:
#        tab_str.append(float(c))
#    return tab_str
#    
#tab_sol_anal = recuperation_tableau_sol_anal()

print("Loading the table of the values of the analytic solution...")
tab_sol_anal = np.load("Val_sol_anal_1D.npy")
print("Loaded")

def anal_sol_interpolated(x):
    '''
    Return an approximation of the analytical solution a point x using an interpolation
    '''
    if x==1:
        return 0
    else:
        i = int(x*N_max*P_max)
        return tab_sol_anal[i]

#--------------------------------------------------------------------------

def L2_norm(h, a, b, P_L2):
    h_squared = lambda x : h(x)**2
    return np.sqrt(integrate(h_squared, a, b, P_L2))

def L2_relative_error(g_anal, g, U, a, b, N, P_L2=N*20, R=1.):
    dif_sol_anal_et_app = lambda x : anal_sol_interpolated(x) - approximate_solution(x, U, N, 50, R)
    return L2_norm(dif_sol_anal_et_app, a, b, P_L2)/L2_norm(g_anal, a, b, P_L2)    

#--------------------------------------------------------------------------

L2_relative_error_Tab=[]
    
fig = plt.figure()

plt.xlabel("Logarithme de la taille de la base d'éléments finis")
plt.ylabel("Logarithme de l'erreur relative en norme L2")

# P is the number of points used in the function "integrate" in a interval of the form [t_x(i), t_x(i+1)]
P = 100
# p allows us to evaluate the relative error on a smaller interval
p = 0

X = np.linspace(0, np.exp(-p), 1000)
tab_N = [n for n in range(3,20)]
for N in tab_N:
    print("N = ",N)
    U = assemble_U(N, P)
#    dif_sol_anal_et_app = lambda x : anal_sol_interpolated(x) - approximate_solution(x, U, N, P)
#    L2_relative_error_Tab.append(L2_relative_error(anal_sol_interpolated, dif_sol_anal_et_app, U, 0, np.exp(-p), N))
    plt.plot(X, [approximate_solution(x, U, N, P) for x in X])

#plt.plot(np.log(tab_N), np.log(L2_relative_error_Tab))

#
#Y_analytique = [solution_analytique_interpolee(x) for x in X]
#plt.plot(X, Y_analytique, color='r')

#--------------------------------------------------------------------------


plt.show()
