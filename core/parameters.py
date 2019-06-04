"""
 * Problem data and computation parameters.
"""

import numpy as np

# ---- Problem data ----

def a_per(x):
    return 1 + np.sin(np.pi * x) ** 2

def a(x):
    return a_per(np.log(x))

def f(x):
    return 1.

# ---- Computation parameters ----

N_max = 10000       # Number of intervals in the mesh used to compute the analytic solution
P_max = 500         # Number of sub intervals in each interval used to compute the analytic solution
P_error = 5 * P_max # This is never used!
N = 20   # Number of intervals in the 1D mesh
N_x = 10 # Number of X axis intervals in the 2D mesh
N_y = 11 # Number of Y axis intervals in the 2D mesh
P = 70   # Number of sub intervals in each interval of any mesh

# ------ PGD -----------
nb_iter = 25           # Number of PGD iterations
eps = 1e-13            # Fixed point algorithm precision
max_rand_int = 1000    # Random coefficients picked for the PGD initialization belong to [-max_rand_int, max_rand_int]
