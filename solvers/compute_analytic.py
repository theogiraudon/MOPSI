"""
 * Analytic solution computation and save.

 * Interpolation points are the midpoints of sub-intervals.
"""

import numpy as np
from core.parameters import a, f, N_max, P_max
from core.integrate import rectangle_midpoints

def inv_a(x):
    return 1 / a(x)

def g(x):
    return inv_a(x) * rectangle_midpoints(f, x, 1, N_max, P_max)

def analytic_solution(x, derivative_1):
    return -(rectangle_midpoints(g, x, 1, N_max, P_max) +
             derivative_1 * rectangle_midpoints(inv_a, x, 1, N_max, P_max))

def analytic_derivative(x, derivative_1):
    return g(x) + derivative_1 * inv_a(x)

solution_values = []
derivative_values = []
derivative_1 = - rectangle_midpoints(g, 0, 1, N_max, P_max) / rectangle_midpoints(inv_a, 0, 1, N_max, P_max)

#print("Computation of the analytic solution ", end="", flush=True)
#for i in range(1, N_max * P_max):
#    if i % (N_max * P_max // 10) == 0:
#        print("#", end="", flush=True)
#    solution_values.append(analytic_solution(i / (N_max * P_max), derivative_1))
#
#print("[DONE]")

print("Computation of the analytic derivative ", end="", flush=True)
for i in range(1, N_max * P_max):
    if i % (N_max * P_max // 10) == 0:
        print("#", end="", flush=True)
    derivative_values.append(analytic_derivative(i / (N_max * P_max), derivative_1))

print("[DONE]")

np.save("../data/analytic_solution_{}_{}.npy".format(N_max, P_max), solution_values)
np.save("../data/analytic_derivative_{}_{}.npy".format(N_max, P_max), derivative_values)