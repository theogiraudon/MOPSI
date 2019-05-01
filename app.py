from solvers.compute_analytic import compute_analytic
from displayers.display_sol_der import display
from displayers.display_errors import display_errors

# compute_analytic(solution=False, derivative=True)
# display('PGD', solution=False, derivative=True)
display('spline', 0, 1, solution=False, derivative=True)
# display_errors('PGD', var='N')
