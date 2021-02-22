import re
import numpy as np
from scipy import interpolate

from uf3.representation import knots
from uf3.regression import regularize


class BSplineConfig:
    """
    -Manage knot resolutions and cutoffs
    -Manage basis functions
    -Manage regularization parameters and generate matrices for regression
    -Arrange matrices by chemical interaction
    -Manage feature indices for fixed coefficients
    """
    def __init__(self,
                 chemical_system,
                 r_min_map=None,
                 r_max_map=None,
                 resolution_map=None,
                 knot_spacing='linear',
                 knots_map=None):
        """
        Args:
            chemical_system (uf3.data.composition.ChemicalSystem)
            r_min_map (dict): map of minimum pair distance per interaction.
                If unspecified, defaults to 1.0 for all interactions.
                e.g. {(A-A): 2.0, (A-B): 3.0, (B-B): 4.0}
            r_max_map (dict): map of maximum pair distance per interaction.
                If unspecified, defaults to 6.0 angstroms for all
                interactions, which probably encompasses 2nd-nearest neighbors,
            resolution_map (dict): map of resolution (number of knot intervals)
                per interaction. If unspecified, defaults to 20 for all two-
                body interactions and 5 for three-body interactions.
            knot_spacing (str): "linear" for uniform spacing
                or "lammps" for knot spacing by r^2.
            knots_map (dict): pre-generated map of knots.
                Overrides other settings.
        """
        self.chemical_system = chemical_system
        self.knot_spacing = knot_spacing
        self.knots_map = {}
        self.knot_subintervals = {}
        self.basis_functions = {}
        if r_min_map is None:
            r_min_map = {}
        self.r_min_map = r_min_map
        if r_max_map is None:
            r_max_map = {}
        self.r_max_map = r_max_map
        if resolution_map is None:
            resolution_map = {}
        self.resolution_map = resolution_map
        # Pregenerated knots_map
        if knots_map is not None:
            for pair in self.interactions_map.get(2, []):
                if pair in knots_map:
                    knot_sequence = knots_map[pair]
                    self.knots_map[pair] = knot_sequence
                    self.r_min_map[pair] = knot_sequence[0]
                    self.r_max_map[pair] = knot_sequence[-1]
                    self.resolution_map[pair] = len(knot_sequence) - 7
            for trio in self.interactions_map.get(3, []):
                if trio in knots_map:
                    knot_sequence = knots_map[trio]
                    self.knots_map[trio] = knot_sequence
                    self.r_min_map[trio] = knot_sequence[0]
                    self.r_max_map[trio] = knot_sequence[-1]
                    self.resolution_map[trio] = len(knot_sequence) - 7
        # Default values
        for pair in self.interactions_map.get(2, []):
            self.r_min_map[pair] = self.r_min_map.get(pair, 1.0)
            self.r_max_map[pair] = self.r_max_map.get(pair, 6.0)
            self.resolution_map[pair] = self.resolution_map.get(pair, 20)
        for trio in self.interactions_map.get(3, []):
            self.r_min_map[trio] = self.r_min_map.get(trio, 1.0)
            self.r_max_map[trio] = self.r_max_map.get(trio, 6.0)
            self.resolution_map[trio] = self.resolution_map.get(trio, 5)
        if self.knot_spacing == 'lammps':
            knot_function = knots.generate_lammps_knots
        elif self.knot_spacing == 'linear':
            knot_function = knots.generate_uniform_knots
        elif self.knot_spacing == 'custom':
            pass
        else:
            raise ValueError('Invalid value of knot_spacing:', knot_spacing)
        # supercell cutoff
        self.r_cut = max(list(self.r_max_map.values()))
        # Generate subintervals and basis functions
        for pair in self.interactions_map.get(2, []):
            if pair not in self.knots_map:  # compute knots if not provided
                r_min = self.r_min_map[pair]
                r_max = self.r_max_map[pair]
                n_intervals = self.resolution_map[pair]
                self.knots_map[pair] = knot_function(r_min, r_max, n_intervals)
            subintervals = knots.get_knot_subintervals(self.knots_map[pair])
            self.knot_subintervals[pair] = subintervals
            self.basis_functions[pair] = generate_basis_functions(subintervals)
        for trio in self.interactions_map.get(3, []):
            if trio not in self.knots_map:
                r_min = self.r_min_map[trio]
                r_max = self.r_max_map[trio]
                r_resolution = self.resolution_map[trio]
                self.knots_map[trio] = knot_function(r_min,
                                                     r_max,
                                                     r_resolution)
            subintervals = knots.get_knot_subintervals(self.knots_map[trio])
            self.knot_subintervals[trio] = subintervals
            self.basis_functions[trio] = generate_basis_functions(subintervals)
        self.partition_sizes = self.get_feature_partition_sizes()

    @property
    def degree(self):
        return self.chemical_system.degree

    @property
    def element_list(self):
        return self.chemical_system.element_list

    @property
    def interactions_map(self):
        return self.chemical_system.interactions_map

    def get_regularization_matrix(self,
                                  ridge_map={},
                                  curvature_map={},
                                  **kwargs):
        """
        Args:
            ridge_map (dict): n-body term ridge regularizer strengths.
                default: {1: 1e-4, 2: 1e-6, 3: 1e-5}
            curvature_map (dict): n-body term curvature regularizer strengths.
                default: {1: 0.0, 2: 1e-5, 3: 1e-5}

        Returns:
            combined_matrix (np.ndarray): regularization matrix made up of
                individual matrices per n-body interaction.
        """
        for k in kwargs:
            if k.lower()[0] == 'r':
                ridge_map[int(re.sub('[^0-9]', '', k))] = float(kwargs[k])
            elif k.lower()[0] == 'c':
                curvature_map[int(re.sub('[^0-9]', '', k))] = float(kwargs[k])
        ridge_map = {1: 1e-4, 2: 1e-9, 3: 1e-6, **ridge_map}
        curvature_map = {1: 0.0, 2: 1e-9, 3: 1e-6, **curvature_map}
        # one-body element terms
        n_elements = len(self.chemical_system.element_list)
        matrix = regularize.get_regularizer_matrix(n_elements,
                                                   ridge=ridge_map[1],
                                                   curvature=0.0)
        matrices = [matrix]
        # two- and three-body terms
        for degree in range(2, self.chemical_system.degree + 1):
            r = ridge_map[degree]
            c = curvature_map[degree]
            interactions = self.chemical_system.interactions_map[degree]
            for interaction in interactions:
                size = self.resolution_map[interaction]
                if degree == 2:
                    matrix = regularize.get_regularizer_matrix(size + 3,
                                                               ridge=r,
                                                               curvature=c)
                elif degree == 3:
                    matrix = regularize.get_penalty_matrix_3D(size + 3,
                                                              size + 3,
                                                              size + 3,
                                                              ridge=r,
                                                              curvature=c)
                else:
                    raise ValueError(
                        "Four-body terms and beyond are not yet implemented.")
                matrices.append(matrix)
        combined_matrix = regularize.combine_regularizer_matrices(matrices)
        return combined_matrix

    def get_feature_partition_sizes(self):
        """Get partition sizes: one-body, two-body, and three-body terms."""
        partition_sizes = [len(self.chemical_system.element_list)]
        for degree in range(2, self.chemical_system.degree + 1):
            interactions = self.chemical_system.interactions_map[degree]
            for interaction in interactions:
                if degree == 2:
                    size = self.resolution_map[interaction] + 3
                    partition_sizes.append(size)
                elif degree == 3:
                    resolutions = self.resolution_map[interaction]
                    size = np.product([resolutions + 3,
                                       + resolutions + 3,
                                       + resolutions + 3])
                    partition_sizes.append(size)
                else:
                    raise ValueError(
                        "Four-body terms and beyond are not yet implemented.")
        return partition_sizes

    def get_fixed_tuples(self,
                         values=0,
                         one_body=True,
                         upper_bounds=True,
                         lower_bounds=False):
        """
        Args:
            values (float, np.ndarray): value or values of fixed coefficients.
            one_body (bool): whether to return tuples for one-body terms.
            upper_bounds (bool): whether to return tuples for trailing
                coefficient of each interaction.
            lower_bounds (bool): whether to return tuples for leading
                coefficient of each interaction.

        Returns:
            fixed (list): list of tuples of indices and coefficients to fix
                before fitting. Useful for ensuring smooth cutoffs or
                fixing multiplicative coefficients.
                e.g. fix=[(0, 10), (15, 0)] fixes the first coefficient (idx=0)
                to 10 and the sixteenth coefficient (idx=15) to 0.
        """
        partition_sizes = self.get_feature_partition_sizes()
        indices = []
        if one_body:
            indices = list(range(partition_sizes[0]))
        lower_idxs = np.cumsum(partition_sizes)[:-1] + 1
        upper_idxs = np.cumsum(partition_sizes)[1:] - 1
        if lower_bounds:
            indices.extend(lower_idxs)
        if upper_bounds:
            indices.extend(upper_idxs)
        if np.array(values).ndim == 0:
            values = np.ones(len(indices)) * values
        fixed = np.vstack([indices, values]).T
        return fixed.astype(int)


def generate_basis_functions(knot_subintervals):
    """
    Args:
        knot_subintervals (list): list of knot subintervals,
            e.g. from ufpotential.representation.knots.get_knot_subintervals

    Returns:
        basis_functions (list): list of scipy B-spline basis functions.
    """
    n_splines = len(knot_subintervals)
    basis_functions = []
    for idx in range(n_splines):
        # loop over number of basis functions
        b_knots = knot_subintervals[idx]
        bs = interpolate.BSpline.basis_element(b_knots, extrapolate=False)
        basis_functions.append(bs)
    return basis_functions


def evaluate_basis_functions(points, basis_functions, nu=0, flatten=True):
    """
    Evaluate basis functions.

    Args:
        points (np.ndarray): vector of points to sample, e.g. pair distances
        basis_functions (list): list of callable basis functions.
        nu (int): compute n-th derivative of basis function. Default 0.
        flatten (bool): whether to flatten values per spline.

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
    n_splines = len(basis_functions)
    values_per_spline = []
    for idx in range(n_splines):
        # loop over number of basis functions
        bspline_values = basis_functions[idx](points, nu=nu)
        bspline_values[np.isnan(bspline_values)] = 0
        values_per_spline.append(bspline_values)
    if not flatten:
        return values_per_spline
    value_per_spline = np.array([np.sum(values)
                                 for values in values_per_spline])
    return value_per_spline


def featurize_force_2B(basis_functions, distances, drij_dR, knot_sequence):
    """
    Args:
        drij_dR (np.ndarray): distance-derivatives, e.g. from
            ufpotential.data.two_body.derivatives_by_interaction.
            Shape is (n_atoms, 3, n_distances).
        distances (np.ndarray): vector of distances of the same length as
            the last dimension of drij_dR.
        basis_functions (list): list of callable basis functions.

    Returns:
        x (np.ndarray): rotation-invariant representations generated
            using BSpline basis corresponding to force information.
            Array shape is (n_atoms, 3, n_basis_functions), where the
            second dimension corresponds to the three cartesian directions.
    """
    n_splines = len(basis_functions)
    n_atoms, _, n_distances = drij_dR.shape
    x = np.zeros((n_atoms, 3, n_splines))
    for bspline_idx in np.arange(n_splines):
        # loop over number of basis functions
        basis_function = basis_functions[bspline_idx]
        b_knots = knot_sequence[bspline_idx: bspline_idx+5]
        mask = np.logical_and(distances > b_knots[0],
                              distances < b_knots[-1])
        # first derivative
        bspline_values = basis_function(distances[mask], nu=1)
        # mask position deltas by distances
        deltas = drij_dR[:, :, mask]
        # broadcast multiplication over atomic and cartesian axis dimensions
        x_splines = np.multiply(bspline_values, deltas)
        x_splines = np.sum(x_splines, axis=-1)
        x[:, :, bspline_idx] = x_splines
    x = -x
    return x


def fit_spline_1d(x, y, knot_sequence):
    """
    Utility function for fitting spline coefficients to a sampled 1D function.
        Useful for comparing fit coefficients against true pair potentials.

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
        # loop over knots to ensure each interval has at least one point
        midpoint = 0.5 * (unique_knots[i] + unique_knots[i+1])
        if x_min > unique_knots[i]:  # pad with zeros below lower-bound
            x = np.insert(x, 0, midpoint)
            y = np.insert(y, 0, y_min)
        elif x_max < unique_knots[i]:  # pad with zeros above upper-bound
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
    lsq = interpolate.LSQUnivariateSpline(x,
                                          y,
                                          knot_sequence,
                                          bbox=(b_min, b_max))
    coefficients = lsq.get_coeffs()
    return coefficients


def find_spline_indices(points, knot_sequence):
    """
    Identify basis functions indices that are non-zero at each point.

    Args:
        points (np.ndarray): list of points.
        knot_sequence (np.ndarray): knot sequence vector.

    Returns:
        points (np.ndarray): array of points repeated four times
        idx (np.ndarray): corresponding basis function index for each
            point (four each).
    """
    # identify basis function "center" per point
    idx = np.searchsorted(np.unique(knot_sequence), points, side='left') - 1
    # tile to identify four non-zero basis functions per point
    offsets = np.tile([0, 1, 2, 3], len(points))
    idx = np.repeat(idx, 4) + offsets
    points = np.repeat(points, 4)
    return points, idx
