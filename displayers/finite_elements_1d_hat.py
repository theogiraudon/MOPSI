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

def analytic_solution(x, solution_values):
    '''
    Interpolate the analytic solution in x.
    '''
    i = int(x * N_max * P_max)
    if i >= len(solution_values):
        return 0
    return solution_values[i]

def analytic_derivative(x, derivative_values):
    '''
    Interpolate the analytic derivative in x.
    '''
    i = int(x * N_max * P_max)
    if i >= len(derivative_values):
        return derivative_values[-1]
    return derivative_values[i]

# ---- 1D Hat errors ----

def L2_relative_solution_error(begin, end, N, P, U, solution_values):
    loaded_analytic_solution = lambda x : analytic_solution(x, solution_values)
    solution_gap = lambda x : loaded_analytic_solution(x) - approximate_solution(x, U, N)
    return L2_norm(solution_gap, begin, end, N, P) / L2_norm(loaded_analytic_solution, begin, end, N, P)

def L2_relative_derivative_error(begin, end, N, P, U, derivative_values):
    loaded_analytic_derivative = lambda x : analytic_derivative(x, derivative_values)
    derivative_gap = lambda x : loaded_analytic_derivative(x) - approximate_derivative(x, U, N)
    return L2_norm(derivative_gap, begin, end, N, P) / L2_norm(loaded_analytic_derivative, begin, end, N, P)

# ---- Solution and derivative display ----

def display_1d_hat(solution=True, derivative=True):
    if solution or derivative:
        U = assemble_U(N, P)
        fig = plt.figure()
        if solution:
            print("Loading analytic solution values...", end="")
            solution_values = np.load("data/analytic_solution_{}_{}.npy".format(N_max, P_max))
            print("Loaded")
            X = np.linspace(0, 1, solution_values.shape[0], endpoint=False)
            plt.plot(X, [analytic_solution(x, solution_values) for x in X])
            plt.plot(X, [approximate_solution(x, U, N) for x in X])
            plt.show()
        if derivative:
            print("Loading analytic derivative values...", end="")
            derivative_values = np.load("data/analytic_derivative_{}_{}.npy".format(N_max, P_max))
            print("Loaded")
            X = np.linspace(0, 1, 100000, endpoint=False)
            plt.plot(X, [analytic_derivative(x, derivative_values) for x in X])
            plt.plot(X, [approximate_derivative(x, U, N) for x in X])
            plt.show()

# ---- Error display ----

def display_1d_hat_errors(solution=True, derivative=True):
    fig = plt.figure()
    nb_N = 5
    max_N = 100
    p_list = [p for p in range(4)] # Errors will be displayed in the [0, e^{-p}] interval.
    N_list = [int(np.exp(k) / np.exp(nb_N) * max_N) + 2 for k in range(2, nb_N + 1)] # N needs to be above 2.

    # Assemble U with increasing values of N.
    U_list = []
    for N in N_list:
        t1 = time()
        U = assemble_U(N, P)
        t2 = time()
        print("Computation time (N = {}, P = {}) : {} seconds".format(N, P, t2 - t1))
        U_list.append(U)

    if solution:
        print("Loading analytic solution values...", end="")
        solution_values = np.load("data/analytic_solution_{}_{}.npy".format(N_max, P_max))
        print("Loaded")
        for p in p_list:
            L2_relative_solution_errors = []
            for N_index in range(len(N_list)):
                L2_relative_solution_errors.append(
                    L2_relative_solution_error(
                        0,
                        np.exp(-p),
                        N_list[N_index],
                        P_error,
                        U_list[N_index],
                        solution_values
                    )
                )
            print("Error display interval : [0, e^-{}]".format(p))
            plt.title("Relative solution error (p = {})".format(p))
            plt.xlabel("Step (log)")
            plt.ylabel("Error L^2 norm (log)")
            plt.plot(-np.log(N_list), np.log(L2_relative_solution_errors))
            plt.show()

    if derivative:
        print("Loading analytic derivative values...", end="")
        derivative_values = np.load("data/analytic_derivative_{}_{}.npy".format(N_max, P_max))
        print("Loaded")
        for p in p_list:
            L2_relative_derivative_errors = []
            for N_index in range(len(N_list)):
                L2_relative_derivative_errors.append(
                    L2_relative_derivative_error(
                        0,
                        np.exp(-p),
                        N_list[N_index],
                        P_error,
                        U_list[N_index],
                        derivative_values
                    )
                )
            print("Error display interval : [0, e^-{}]".format(p))
            plt.title("Relative derivative error (p = {})".format(p))
            plt.xlabel("Step (log)")
            plt.ylabel("Error L^2 norm (log)")
            plt.plot(-np.log(N_list), np.log(L2_relative_derivative_errors))
            plt.show()