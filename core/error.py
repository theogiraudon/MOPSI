"""
 * Error computation.

 * Interpolation points are the midpoints of sub-intervals.
"""

import numpy as np

from scipy.integrate import quad
from core.integrate import rectangle_midpoints

def L2_norm(h, begin, end, N, P):
    h_squared = lambda x : h(x)**2
    # return np.sqrt(rectangle_midpoints(h_squared, begin, end, N, P))
    return np.sqrt(quad(h_squared, begin, end)[0])