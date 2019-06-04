from solvers.compute_analytic import compute_analytic
from displayers.display_sol_der import display
from displayers.display_errors import display_errors
from solvers.finite_elements_PGD import display_errors_PGD

# Don't uncomment the following line unless you want to generate again the analytic_derivative_N_max_P_max.npy file
# containing the analytic derivative. The .npy is created inside the "data" folder. The computation of the
# analytic solution is much longer than the analytic derivative because it involves much more integration.

# compute_analytic(solution=False, derivative=True)



# The following function displays the derivative for each method : the first argument can be 'hat', 'spline', '2D'
# or 'PGD'. The derivative will be computed with the parameters which are in the file core/parameters.py : you
# can change them.
# Here the derivative is shown on [0,1] but you can plot it in any sub-interval of [0,1] you want.

# display('spline', 0, 1, solution=False, derivative=True)


# The following function displays the log error in [0,1], [0, exp(-1)], [0, exp(-2)] and [0, exp(-3)] for 'hat',
# 'spline' and '2d'.
# You can choose to display the log error as a function of -log(N) or -log(P).
# The two numerical arguments are used to produce the list of N, which is of the form
# [exp(alpha*i) + 4 for i in range(3, end + 1)].

# display_errors('hat', 0.22, 30, var='N')
# display_errors('2D', 0.22, 15, var='N')


# The following function displays the log error in [0,1] as a function of the number n of iterations for the PGD
# method. You can specify the first n, the last n and the step between the n.

# display_errors_PGD(2,20,2)

