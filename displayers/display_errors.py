import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from time import time

from solvers.finite_elements_1d_hat import assemble_U, approximate_derivative
from solvers.finite_elements_1d_spline import assemble_U_spline, approximate_derivative_spline
from solvers.finite_elements_2d import assemble_U, approximate_derivative_2D
from solvers.compute_analytic import analytic_derivative
from core.parameters import N_max, P_max, P_error, N, P
from core.error import L2_norm

print("Number of main mesh intervals (analytic) : N_max = {}".format(N_max))
print("Number of sub intervals (analytic) : P_max = {}".format(P_max))

def L2_relative_derivative_error(ef, begin, end, N0, P0, U, derivative_values):
    loaded_analytic_derivative = lambda x : analytic_derivative(x, derivative_values)
    if ef == 'hat':
        derivative_gap = lambda x : loaded_analytic_derivative(x) - approximate_derivative(x, U, N0)
    elif ef == 'spline':
        derivative_gap = lambda x : loaded_analytic_derivative(x) - approximate_derivative_spline(x, U, N0)
    elif ef == '2D':
        derivative_gap = lambda x : loaded_analytic_derivative(x) - approximate_derivative_2D(x, np.log(x), U, N0, N0+1)
    return L2_norm(derivative_gap, begin, end, N0, P0) / L2_norm(loaded_analytic_derivative, begin, end, N0, P0)


# We only display the derivative error
def display_errors(ef, var='N'):
    fig = plt.figure()
    p_list = range(4) # Errors will be displayed in the [0, e^{-p}] interval.

    if var=='N':
        nb_N = 15

        N_list = [int(np.exp(0.25*k)) + 2 for k in range(3, nb_N + 1)]
        print(max(N_list))
        N_list = np.sort(list(set(N_list))) # We remove the repetition of N
        N_list = [int(N_index) for N_index in N_list] # We transform the list of float in a list of int
        print(N_list)
        # Assemble U with increasing values of N.
        U_list = []
        for N_index in N_list:
            t1 = time()
            if ef == 'hat':
                U = assemble_U(N_index, P)
            elif ef == 'spline':
                U = assemble_U_spline(N_index, P)
            elif ef == '2D':
                U = assemble_U(N_index, N_index + 1, P)
            t2 = time()
            print("U computation time (N = {}, P = {}) : {} seconds".format(N_index, P, t2 - t1),flush=True)
            U_list.append(U)

        print("Loading analytic derivative values...", end="",flush=True)
        derivative_values = np.load("data/analytic_derivative_{}_{}.npy".format(N_max, P_max))
        print("Loaded")
        for p in p_list:
            L2_relative_derivative_errors = []
            for N_index in range(len(N_list)):
                t1 = time()
                L2_relative_derivative_errors.append(
                    L2_relative_derivative_error(
                        ef,
                        0,
                        np.exp(-p),
                        N_list[N_index],
                        P_error,
                        U_list[N_index],
                        derivative_values
                    )
                )
                t2 = time()
                print("Error computation time (N = {}, P = {}) : {} seconds".format(N_list[N_index], P, t2 - t1), flush=True)
            print("Error display interval : [0, e^-{}]".format(p))
            H = [-np.log(n+1) for n in N_list]
            plt.plot(H, np.log(L2_relative_derivative_errors))
            plt.savefig('images/Derivative_errorN_p{}_P{}_{}.png'.format(p, P, ef))
            plt.show()

            # Linear regression to get the slope of our straight line
            slope, intercept, r_value, p_value, std_err = stats.linregress(H, np.log(L2_relative_derivative_errors))
            print("For p = {}, the L^2 error increases as the step power {}".format(p, slope))

    elif var=='P':
        nb_P = 70
        max_P = 250

        P_list = [int(np.exp(0.1*k)) + 2 for k in range(2, nb_P + 1)]  # P needs to be above 2.
        P_list = np.sort(list(set(P_list)))  # We remove the repetition of P
        P_list = [int(P_index) for P_index in P_list]

        # Assemble U with increasing values of N.
        U_list = []
        for P_index in P_list:
            t1 = time()
            if ef == 'hat':
                U = assemble_U(N, P_index)
            elif ef == 'spline':
                U = assemble_U_spline(N, P_index)
            elif ef == '2D':
                U = assemble_U(N, N + 1, P_index)
            t2 = time()
            print("Computation time (N = {}, P = {}) : {} seconds".format(N, P_index, t2 - t1), flush=True)
            U_list.append(U)

        print("Loading analytic derivative values...", end="", flush=True)
        derivative_values = np.load("data/analytic_derivative_{}_{}.npy".format(N_max, P_max))
        print("Loaded")
        for p in p_list:
            L2_relative_derivative_errors = []
            for P_index in range(len(P_list)):
                t1 = time()
                L2_relative_derivative_errors.append(
                    L2_relative_derivative_error(
                        ef,
                        0,
                        np.exp(-p),
                        N,
                        P_error,
                        U_list[P_index],
                        derivative_values
                    )
                )
                t2 = time()
                print("Error computation time (N = {}, P = {}) : {} seconds".format(N, P_list[P_index], t2 - t1),
                      flush=True)
            print("Error display interval : [0, e^-{}]".format(p))
            H = [-np.log(P) for P in P_list]
            plt.plot(H, np.log(L2_relative_derivative_errors))
            plt.savefig('images/Derivative_errorP_p{}_N{}_{}.png'.format(p, N, ef))
            plt.show()

            # Linear regression to get the slope of our straight line
            slope, intercept, r_value, p_value, std_err = stats.linregress(H, np.log(L2_relative_derivative_errors))
            print("For p = {}, the L^2 error increases as the step power {}".format(p, slope))