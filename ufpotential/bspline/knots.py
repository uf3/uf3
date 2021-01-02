import numpy as np


def knot_sequence_from_points(knot_points):
    """
    Repeat endpoints to satisfy knot sequence requirements (i.e. fixing first
        and second derivatives to zero).

    Args:
        knot_points (list or numpy.ndarray): sorted knot points in
            increasing order.

    Returns:
        knots (numpy.ndarray): knot sequence with repeated ends.
    """
    knots = np.concatenate([np.repeat(knot_points[0], 3),
                            knot_points,
                            np.repeat(knot_points[-1], 3)])
    return knots


def get_knot_subintervals(knots):
    """
    Generate 5-knot subintervals for individual basis functions
        from specified knot sequence.

    Args:
        knots (numpy.ndarray): knot sequence with repeated ends.

    Returns:
        subintervals (list): list of 5-knot subintervals.
    """
    subintervals = [knots[i:i+5]
                    for i in range(len(knots)-4)]
    return subintervals


def generate_uniform_knots(r_min, r_max, n_intervals, sequence=True):
    """
    Generate evenly-spaced knot points or knot sequence.
    
    Args:
        r_min (float): lower-bound for knot points.
        r_max (float): upper-bound for knot points.
        n_intervals (int): number of unique intervals in the knot sequence,
            i.e. n_intervals + 1 samples will be taken between r_min and r_max.
        sequence (bool): whether to repeat ends to yield knot sequence.

    Returns:
        knots (numpy.ndarray): knot points or knot sequence.
    """
    knots = np.linspace(r_min, r_max, n_intervals + 1)
    if sequence:
        knots = knot_sequence_from_points(knots)
    return knots