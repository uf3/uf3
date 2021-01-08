import numpy as np
from scipy.interpolate import BSpline
from scipy.interpolate import LSQUnivariateSpline


def evaluate_bspline(points, knot_subintervals, flatten=True):
    """
    Args:
        points (np.ndarray): vector of points to sample, e.g. pair distances
        knot_subintervals (list): list of knot subintervals,
            e.g. from ufpotential.representation.knots.get_knot_subintervals

    Returns:
        if flatten:
            value_per_spline (np.ndarray): vector of cubic B-spline value,
                summed across queried points, for each knot subinterval.
                Used as a rotation-invariant representation generated
                using a BSpline basis.
        else:
            values_per_spline (list): list of vector of cubic B-spline
                evaluations for each knot subinterval.
    """
    n_splines = len(knot_subintervals)
    values_per_spline = []
    for idx in range(n_splines):
        # loop over number of basis functions
        b_knots = knot_subintervals[idx]
        bs_l = BSpline.basis_element(b_knots, extrapolate=False)
        mask = np.logical_and(points >= b_knots[0],
                              points <= b_knots[4])
        bspline_values = bs_l(points[mask])
        values_per_spline.append(bspline_values)
    if not flatten:
        return values_per_spline
    value_per_spline = np.array([np.sum(values)
                                for values in values_per_spline])
    return value_per_spline


def compute_force_bsplines(drij_dR, distances, knot_intervals):
    """
    Args:
        drij_dR (np.ndarray): distance-derivatives, e.g. from
            ufpotential.data.two_body.derivatives_by_interaction.
            Shape is (n_atoms, 3, n_distances).
        distances (np.ndarray): vector of distances of the same length as
            the last dimension of drij_dR.
        knot_intervals (list): list of knot subintervals,
            e.g. from ufpotential.representation.knots.get_knot_subintervals

    Returns:
        x (np.ndarray): rotation-invariant representations generated
            using BSpline basis corresponding to force information.
            Array shape is (n_atoms, n_basis_functions, 3), where the
            last dimension corresponds to the three cartesian directions.
    """
    n_splines = len(knot_intervals)
    n_atoms, _, n_distances = drij_dR.shape
    x = np.zeros((n_splines, n_atoms, 3))
    for idx in np.arange(n_splines):
        # loop over number of basis functions
        x_splines = np.zeros((n_atoms, 3))
        b_knots = knot_intervals[idx]
        bs_l = BSpline.basis_element(b_knots, extrapolate=False)
        mask = np.logical_and(distances > b_knots[0],
                              distances < b_knots[4])
        bspline_values = bs_l(distances[mask], nu=1)  # first derivative
        for m_idx, atom_deltas in enumerate(drij_dR):
            # loop over atoms
            x_cartesian = np.zeros(3)
            for l_idx, cartesian_deltas in enumerate(atom_deltas):
                # loop over cartesian directions
                deltas = cartesian_deltas[mask]
                x_cartesian[l_idx] = np.sum(np.multiply(bspline_values,
                                                        deltas))
            x_splines[m_idx, :] = x_cartesian
        x[idx, :, :] = x_splines
    x = -x.transpose(1, 2, 0)
    return x


def fit_spline_1d(x, y, knot_sequence):
    """
    Utility function for fitting spline coefficients to a sampled 1D function.

    Args:
        x (np.ndarray): vector of function inputs.
        y (np.ndarray): vector of corresponding function outputs.
        knot_sequence (np.ndarray): knot sequence,
            e.g. from ufpotential.knots.generate_lammps_knots

    Returns:
        coefficients (np.ndarray): vector of cubic B-spline coefficients.
    """
    # scipy requirement: data must not lie outside of knot range
    b_min = knot_sequence[0]
    b_max = knot_sequence[-1]
    y = y[(x > b_min) & (x < b_max)]
    x = x[(x > b_min) & (x < b_max)]
    # scipy requirement: knot intervals must include at least 1 point each
    lowest_idx = np.argmin(x)
    highest_idx = np.argmax(x)
    x_min = x[lowest_idx]
    y_min = y[lowest_idx]
    x_max = x[highest_idx]
    y_max = y[highest_idx]
    unique_knots = np.unique(knot_sequence)
    n_knots = len(unique_knots)
    for i in range(n_knots-1):
        midpoint = 0.5 * (unique_knots[i] + unique_knots[i+1])
        if x_min > unique_knots[i]:
            x = np.insert(x, 0, midpoint)
            y = np.insert(y, 0, y_min)
        elif x_max < unique_knots[i]:
            x = np.insert(x, -1, midpoint)
            y = np.insert(y, -1, y_max)
    # scipy requirement: samples must be in increasing order
    x_sort = np.argsort(x)
    x = x[x_sort]
    y = y[x_sort]
    if knot_sequence[0] == knot_sequence[3]:
        knot_sequence = knot_sequence[4:-4]
    else:
        knot_sequence = knot_sequence[1:-1]
    lsq = LSQUnivariateSpline(x, y, knot_sequence, bbox=(b_min, b_max))
    coefficients = lsq.get_coeffs()
    return coefficients
