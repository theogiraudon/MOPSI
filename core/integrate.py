"""
 * Rectangle and trapezoidal rules on a 1D mesh.

 * Interpolation points may be segment edges or midpoints.
"""

def rectangle_midpoints(h, begin, end, N, P):
    '''
    Integrate h in the [begin, end] range using the rectangle rule.

    The [0, 1] range is split into N intervals.
    Each of these N intervals comprises P intervals.
    Interpolation points are the midpoints of these sub-intervals.

    :param h:     Function to integrate.
    :param begin: Minimum of the integration interval.
    :param end:   Maximum of the integration interval.
    :param N:     Number of intervals in [0, 1].
    :param P:     Number of sub-intervals in each interval.
    '''
    step = 1 / (N * P) # Step of the mesh.
    if begin < 0:
        begin = 0
    begin_point = ((begin + 1 / (2 * N * P)) // step + 1 / 2) * step # Smallest interpolation point in [begin, end].
    end_point = ((end + 1 / (2 * N * P)) // step - 1 / 2) * step     # Greatest interpolation point in [begin, end].
    # Integration interval length is preserved.
    result = sum([step * h(begin_point + i * step) for i in range(int(round((end_point - begin_point) / step)) + 1) ])
    return result