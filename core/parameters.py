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

N_max = 10000
P_max = 500
P_error = 5 * P_max
N=50
N_x = 30
N_y = 31
P = 70

# ------ PGD -----------
nb_iter = 30           # Number of PGD iterations
eps = 1e-16            # Fixed point algorithm precision
max_rand_int = 1000    # Random coefficients picked for the PGD initialization belong to [-max_rand_int, max_rand_int]
