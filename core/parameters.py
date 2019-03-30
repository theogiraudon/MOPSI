"""
 * Problem data and computation parameters.
"""

import numpy as np

# ---- Problem data ----

def a_per(x):
    return 1 + np.sin(2 * np.pi * x) ** 2

def a(x):
    return a_per(np.log(x))

def f(x):
    return 1.

# ---- Computation parameters ----

# N_max = 500
# P_max = 200
N_max = 10000
P_max = 500
P_error = 5 * P_max
N = 2000
P = 200