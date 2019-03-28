"""
 * 1D hat functions results display.
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time

from solvers.finite_elements_1d_hat import assemble_U, approximate_solution, approximate_derivative
from core.parameters import N_max, P_max, P_error, N, P
from core.error import L2_norm

print("Number of main mesh intervals (analytic) : N_max = {}".format(N_max))
print("Number of sub intervals (analytic) : P_max = {}".format(P_max))

# ---- Analytic data loading ----

print("Loading analytic solution values...", end="")
analytic_values = np.load("../data/analytic_solution_{}_{}.npy".format(N_max, P_max))
print("Loaded")

print("Loading analytic derivative values...", end="")
derivative_values = np.load("../data/analytic_derivative_{}_{}.npy".format(N_max, P_max))
print("Loaded")

def analytic_solution(x):
    '''
    Interpolate the analytic solution in x.
    '''
    i = int(x * N_max * P_max)
    if i >= len(analytic_values):
        return 0
    return analytic_values[i]

def analytic_derivative(x):
    '''
    Interpolate the analytic derivative in x.
    '''
    i = int(x * N_max * P_max)
    if i >= len(analytic_values):
        return derivative_values[-1]
    return derivative_values[i]

# ---- 1D Hat errors ----

def L2_relative_solution_error(begin, end, N, P, U):
    solution_gap = lambda x : analytic_solution(x) - approximate_solution(x, U, N)
    return L2_norm(solution_gap, begin, end, N, P) / L2_norm(analytic_solution, begin, end, N, P)

def L2_relative_derivative_error(begin, end, N, P, U):
    derivative_gap = lambda x : analytic_derivative(x) - approximate_derivative(x, U, N)
    return L2_norm(derivative_gap, begin, end, N, P) / L2_norm(analytic_derivative, begin, end, N, P)

# ---- Solution and derivative display ----

# U = assemble_U(N, P)
# fig = plt.figure()
# X = np.linspace(0, 1, analytic_values.shape[0], endpoint=False)
# plt.plot(X, [analytic_solution(x) for x in X])
# plt.plot(X, [approximate_solution(x, U, N) for x in X])
# plt.show()
# plt.plot(X, [analytic_derivative(x) for x in X])
# plt.plot(X, [approximate_derivative(x, U, N) for x in X])
# plt.show()

# ---- Error display ----

nb_N = 5
max_N = 100
p_list = [p for p in range(4)] # Errors will be displayed in the [0, e^{-p}] interval.
N_list = [int(np.exp(k) / np.exp(nb_N) * max_N) for k in range(2, nb_N + 1)]
print(N_list)

# Assemble U with increasing values of N.
U_list = []
for N in N_list:
    t1 = time()
    U = assemble_U(N, P)
    t2 = time()
    print("Computation time (N = {}, P = {}) : {} seconds".format(N, P, t2 - t1))
    U_list.append(U)

for p in p_list:
    L2_relative_solution_errors = []
    L2_relative_derivative_errors = []
    for N_index in range(len(N_list)):
        L2_relative_solution_errors.append(
            L2_relative_solution_error(0, np.exp(-p), N_list[N_index], P, U_list[N_index])
        )
        L2_relative_derivative_errors.append(
            L2_relative_derivative_error(0, np.exp(-p), N_list[N_index], P, U_list[N_index])
        )
    print("Error display interval : [0, e^-{}]".format(p))
    # Solution error display.
    fig = plt.figure()
    plt.subplot(211)
    plt.title("Solution error (p = {})".format(p))
    plt.xlabel("Step (log)")
    plt.ylabel("Error L^2 norm (log)")
    plt.plot(-np.log(N_list), np.log(L2_relative_solution_errors))
    # Derivative error display.
    plt.subplot(212)
    plt.title("Derivative error (p = {})".format(p))
    plt.xlabel("Step (log)")
    plt.ylabel("Derivative error L^2 norm (log)")
    plt.plot(-np.log(N_list), np.log(L2_relative_derivative_errors))
    plt.tight_layout()
    plt.show()