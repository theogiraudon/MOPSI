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
N = 1000
N_x = 30
N_y = 30
P = 70