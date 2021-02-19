import numpy as np
from uf3.data import composition
from uf3.util import json_io


def knot_sequence_from_points(knot_points):
    """
    Repeat endpoints to satisfy knot sequence requirements (i.e. fixing first
        and second derivatives to zero).

    Args:
        knot_points (list or np.ndarray): sorted knot points in
            increasing order.

    Returns:
        knots (np.ndarray): knot sequence with repeated ends.
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
        knots (np.ndarray): knot sequence with repeated ends.

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
        knots (np.ndarray): knot points or knot sequence.
    """
    knots = np.linspace(r_min, r_max, n_intervals + 1)
    if sequence:
        knots = knot_sequence_from_points(knots)
    return knots


def generate_lammps_knots(r_min, r_max, n_intervals, sequence=True):
    """
    Generate knot points or knot sequence using LAMMPS convention of
    distance^2. This scheme yields somewhat higher resolution at larger
    distances and somewhat lower resolution at smaller distances.
    Since speed is mostly unaffected by the number of basis functions, due
    to the local support, a high value of n_intervals ensures resolution
    while ensuring expected behavior in LAMMPS.

    Args:
        r_min (float): lower-bound for knot points.
        r_max (float): upper-bound for knot points.
        n_intervals (int): number of unique intervals in the knot sequence,
            i.e. n_intervals + 1 samples will be taken between r_min and r_max.
        sequence (bool): whether to repeat ends to yield knot sequence.

    Returns:
        knots (np.ndarray): knot points or knot sequence.
    """
    knots = np.linspace(r_min ** 2, r_max ** 2, n_intervals + 1) ** 0.5
    if sequence:
        knots = knot_sequence_from_points(knots)
    return knots


def parse_knots_file(filename, chemical_system):
    """
    Args:
        filename (str)
        chemical_system (composition.ChemicalSystem)

    Returns:
        knots_map (dict): map of knots per chemical interaction.
    """
    json_data = json_io.load_interaction_map(filename)
    knots_map = {}
    for d in range(2, chemical_system.degree + 1):
        for interaction in chemical_system.interactions_map[d]:
            if interaction in json_data:
                array = json_data[interaction]
                conditions = [np.ptp(array[:4]) == 0,
                              np.ptp(array[-4:]) == 0,
                              np.all(np.gradient(array) >= 0)]
                if all(conditions):
                    knots_map[interaction] = array
    return knots_map
