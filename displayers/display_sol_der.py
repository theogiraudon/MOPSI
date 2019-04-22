import numpy as np
import matplotlib.pyplot as plt
from time import time

from solvers.finite_elements_1d_hat import assemble_U, approximate_solution, approximate_derivative
from solvers.finite_elements_1d_spline import assemble_U_spline, approximate_solution_spline, approximate_derivative_spline
from solvers.finite_elements_2d import assemble_U_2D, approximate_solution_2D, approximate_derivative_2D
from solvers.compute_analytic import analytic_derivative, analytic_solution
from core.parameters import N_max, P_max, N, N_x, N_y, P


def display(ef, solution=True, derivative=True):
    if solution or derivative:
        t0 = time()
        print("Assembling the matrix U...")
        if ef=='hat':
            U = assemble_U(N, P)
        elif ef=='spline':
            U = assemble_U_spline(N, P)
        elif ef=='2D':
            U = assemble_U_2D(N_x, N_y, P)
        print("Matrix U assembled. \n")
        print("Time used to assemble U : {}".format(time() - t0))

        fig = plt.figure()
        if solution:
            print("Loading analytic solution values...", end="")
            solution_values = np.load("data/analytic_solution_{}_{}.npy".format(N_max, P_max))
            print("Loaded")
            X = np.linspace(0, 1, solution_values.shape[0], endpoint=False)
            plt.plot(X, [analytic_solution(x, solution_values) for x in X])
            if ef == 'hat':
                plt.plot(X, [approximate_solution(x, U, N) for x in X])
            elif ef == 'spline':
                plt.plot(X, [approximate_solution_spline(x, U, N) for x in X])
            elif ef == '2D':
                plt.plot(X[1:-1], [approximate_solution_2D(x, np.log(x), U, N_x, N_y) for x in X[1:-1]])
            plt.show()
        if derivative:
            print("Loading analytic derivative values...", end="")
            derivative_values = np.load("data/analytic_derivative_{}_{}.npy".format(N_max, P_max))
            print("Loaded")
            X = np.linspace(0, 1, 2000, endpoint=False)
            plt.plot(X, [analytic_derivative(x, derivative_values) for x in X])
            if ef == 'hat':
                plt.plot(X, [approximate_derivative(x, U, N) for x in X])
            elif ef == 'spline':
                plt.plot(X, [approximate_derivative_spline(x, U, N) for x in X])
            elif ef == '2D':
                plt.plot(X[1:-1], [approximate_derivative_2D(x, np.log(x), U, N_x, N_y) for x in X[1:-1]])
            plt.show()