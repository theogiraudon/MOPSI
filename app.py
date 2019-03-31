from solvers.compute_analytic import compute_analytic
from displayers.finite_elements_1d import display_1d, display_1d_errors

# compute_analytic(solution=False, derivative=True)
# display_1d('hat', solution=False, derivative=True)
display_1d('spline', solution=False, derivative=True)
# display_1d_errors('spline', solution=False, derivative=True,var='N')
