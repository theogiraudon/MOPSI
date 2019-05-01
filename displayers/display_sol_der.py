import numpy as np
import matplotlib.pyplot as plt
from time import time

from solvers.finite_elements_1d_hat import assemble_U, approximate_solution, approximate_derivative
from solvers.finite_elements_1d_spline import assemble_U_spline, approximate_solution_spline, approximate_derivative_spline
from solvers.finite_elements_2d import assemble_U, approximate_solution_2D, approximate_derivative_2D
from solvers.finite_elements_PGD import approximate_U_PGD, approximate_U_derivative_PGD, PGD
from solvers.compute_analytic import analytic_derivative, analytic_solution
from core.parameters import N_max, P_max, N, N_x, N_y, P


def display(ef, begin=0, end=0, solution=True, derivative=True):
    if solution or derivative:
        t0 = time()
        print("Assembling the matrix U...")
        if ef=='hat':
            U = assemble_U(N, P)
        elif ef=='spline':
            U = assemble_U_spline(N, P)
        elif ef=='2D':
            U = assemble_U(N_x, N_y, P)
        elif ef=='PGD':
            R_list1, S_list1 = PGD(N_x, N_y)

        print("Matrix U assembled. \n")
        print("Time used to assemble U : {}".format(time() - t0))

        fig = plt.figure()
        if solution:
            print("Loading analytic solution values...", end="")
            solution_values = np.load("data/analytic_solution_{}_{}_{}.npy".format(N_max, P_max, ef))
            print("Loaded")
            X = np.linspace(begin, end, solution_values.shape[0], endpoint=False)
            plt.plot(X, [analytic_solution(x, solution_values) for x in X])
            if ef == 'hat':
                plt.plot(X, [approximate_solution(x, U, N) for x in X])
            elif ef == 'spline':
                plt.plot(X, [approximate_solution_spline(x, U, N) for x in X])
            elif ef == '2D':
                plt.plot(X[1:-1], [approximate_solution_2D(x, np.log(x), U, N_x, N_y) for x in X[1:-1]])
            elif ef == 'PGD':
                plt.plot(X[1:], [approximate_U_PGD(x, np.log(x), R_list1, S_list1, N_x, N_y) for x in X[1:]])
            plt.show()
        if derivative:
            print("Loading analytic derivative values...", end="")
            derivative_values = np.load("data/analytic_derivative_{}_{}.npy".format(N_max, P_max))
            print("Loaded")
            X = np.linspace(begin, end, 3000, endpoint=False)
            plt.plot(X, [analytic_derivative(x, derivative_values) for x in X])
            t0 = time()
            if ef == 'hat':
                plt.plot(X, [approximate_derivative(x, U, N) for x in X])
            elif ef == 'spline':
                plt.plot(X, [approximate_derivative_spline(x, U, N) for x in X])
            elif ef == '2D':
                plt.plot(X[1:-1], [approximate_derivative_2D(x, np.log(x), U, N_x, N_y) for x in X[1:-1]])
            elif ef == 'PGD':
                plt.plot(X[1:], [approximate_U_derivative_PGD(x, np.log(x), R_list1, S_list1, N_x, N_y) for x in X[1:]])
            print("Time used to display : {}".format(time() - t0))
            plt.show()