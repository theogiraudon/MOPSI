from solvers.compute_analytic import compute_analytic
from displayers.finite_elements_1d_hat import display_1d_hat, display_1d_hat_errors

# compute_analytic(solution=False, derivative=True)
display_1d_hat(solution=False, derivative=True)
display_1d_hat_errors(solution=True, derivative=False)