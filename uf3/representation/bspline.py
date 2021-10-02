from typing import List, Dict, Union, Tuple, Any
import os
import re
import warnings
import numpy as np
from scipy import interpolate

from uf3.data import composition
from uf3.representation import angles
from uf3.regression import regularize
from uf3.util import json_io


class BSplineBasis:
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
                 knot_strategy='linear',
                 offset_1b=True,
                 trailing_trim=3,
                 mask_trim=True,
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
            knot_strategy (str): "linear" for uniform spacing
                or "lammps" for knot spacing by r^2.
            knots_map (dict): pre-generated map of knots.
                Overrides other settings.
        """
        self.chemical_system = chemical_system
        self.knot_strategy = knot_strategy
        self.offset_1b = offset_1b
        self.trailing_trim = trailing_trim
        self.mask_trim = mask_trim
        self.r_min_map = {}
        self.r_max_map = {}
        self.resolution_map = {}
        self.knots_map = {}
        self.knot_subintervals = {}
        self.basis_functions = {}
        self.symmetry = {}
        self.flat_weights = {}
        self.template_mask = {}
        self.templates = {}
        self.partition_sizes = []
        self.frozen_c = []
        self.col_idx = []
        self.r_cut = 0.0
        self.update_knots(r_max_map, r_min_map, resolution_map, knots_map)
        self.knot_spacer = get_knot_spacer(self.knot_strategy)
        self.update_basis_functions()

    @staticmethod
    def from_config(config):
        """Instantiate from configuration dictionary"""
        if "chemical_system" not in config:
            raise ValueError("No chemical system specified.")
        chemical_system = config["chemical_system"]
        basis_settings = dict()
        if "knots_path" in config and config["load_knots"]:
            knots_fname = config["knots_path"]
            if os.path.isfile(knots_fname):
                try:
                    knots_json = json_io.load_interaction_map(knots_fname)
                    knots_map = knots_json["knots"]
                except (ValueError, KeyError, IOError):
                    knots_map = None
                basis_settings["knots_map"] = knots_map
        aliases = dict(r_min="r_min_map",
                       r_max="r_max_map",
                       resolution="resolution_map",
                       fit_offsets="offset_1b")
        for key, alias in aliases.items():
            if key in config:
                basis_settings[alias] = config[key]
            if alias in config:  # higher priority in case of duplicate
                basis_settings[alias] = config[alias]
        keys = ["trailing_trim", "mask_trim", "knot_strategy"]
        basis_settings.update({k: v for k, v in config.items() if k in keys})
        bspline_config = BSplineBasis(chemical_system, **basis_settings)
        if "knots_path" in config and config["dump_knots"]:
            knots_map = bspline_config.knots_map
            json_io.dump_interaction_map(dict(knots=knots_map),
                                         filename=config["knots_path"],
                                         write=True)
        return bspline_config

    @property
    def degree(self):
        return self.chemical_system.degree

    @property
    def element_list(self):
        return self.chemical_system.element_list

    @property
    def interactions_map(self):
        return self.chemical_system.interactions_map

    @property
    def interactions(self):
        return self.chemical_system.interactions

    @property
    def n_feats(self) -> int:
        return int(np.sum(self.get_feature_partition_sizes()))

    def __repr__(self):
        summary = ["BSplineBasis:",
                   f"    Basis functions: {self.n_feats}",
                   self.chemical_system.__repr__()
                   ]
        return "\n".join(summary)

    def __str__(self):
        return self.__repr__()

    def get_cutoff(self):
        values = []
        for interaction in self.r_max_map:
            r_max = self.r_max_map[interaction]
            if isinstance(r_max, (float, np.floating, int)):
                values.append(r_max)
            else:
                values.append(r_max[0])
        return max(values)

    def update_knots(self,
                     r_max_map: Dict[Tuple, Any] = None,
                     r_min_map: Dict[Tuple, Any] = None,
                     resolution_map: Dict[Tuple, Any] = None,
                     knots_map: Dict[Tuple, Any] = None):
        # lower and upper distance cutoffs
        if r_min_map is not None:
            r_min_map = composition.sort_interaction_map(r_min_map)
            self.r_min_map.update(r_min_map)
        if r_max_map is not None:
            r_max_map = composition.sort_interaction_map(r_max_map)
            self.r_max_map.update(r_max_map)
        if resolution_map is not None:
            resolution_map = composition.sort_interaction_map(resolution_map)
            self.resolution_map.update(resolution_map)
        # Update with pregenerated knots_map
        if knots_map is not None:
            self.update_knots_from_dict(knots_map)
        # Update with provided and default values
        for pair in self.interactions_map.get(2, []):
            self.r_min_map[pair] = self.r_min_map.get(pair, 1.0)
            self.r_max_map[pair] = self.r_max_map.get(pair, 6.0)
            self.resolution_map[pair] = self.resolution_map.get(pair, 20)
        for trio in self.interactions_map.get(3, []):
            self.r_min_map[trio] = self.r_min_map.get(trio, [1.0, 1.0, 1.0])
            self.r_max_map[trio] = self.r_max_map.get(trio, [4.0, 4.0, 8.0])
            self.resolution_map[trio] = self.resolution_map.get(trio,
                                                                [5, 5, 10])
            min_set = len(set(self.r_min_map[trio]))
            max_set = len(set(self.r_max_map[trio]))
            res_set = len(set(self.resolution_map[trio]))
            if min_set == 1 and max_set == 1 and res_set == 1:
                self.symmetry[trio] = 3
            elif min_set <= 2 and max_set <= 2 and res_set <= 2:
                self.symmetry[trio] = 2
            else:
                self.symmetry[trio] = 1
        self.r_cut = self.get_cutoff()

    def update_knots_from_dict(self, knots_map):
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
                # specified one or more knot sequences
                if isinstance(knot_sequence[0], (float, np.floating, int)):
                    # one knot sequence provided (three-fold symmetry)
                    self.symmetry[trio] = 3
                    l_sequence = knot_sequence
                    m_sequence = knot_sequence
                    n_sequence = knot_sequence
                else:  # zero or one mirror plane
                    if len(knot_sequence) == 2:
                        self.symmetry[trio] = 2
                        l_sequence, n_sequence = knot_sequence
                        m_sequence = l_sequence
                    else:
                        if len(knot_sequence) > 3:
                            warnings.warn(
                                "More than three knot sequences provided "
                                "for {} interaction.".format(trio),
                                RuntimeWarning)
                        self.symmetry[trio] = 1
                        l_sequence = knot_sequence[0]
                        m_sequence = knot_sequence[1]
                        n_sequence = knot_sequence[2]
                self.knots_map[trio] = [l_sequence,
                                        m_sequence,
                                        n_sequence]
                self.r_min_map[trio] = [l_sequence[0],
                                        m_sequence[0],
                                        n_sequence[0]]
                self.r_max_map[trio] = [l_sequence[-1],
                                        m_sequence[-1],
                                        n_sequence[-1]]
                self.resolution_map[trio] = [len(l_sequence) - 7,
                                             len(m_sequence) - 7,
                                             len(n_sequence) - 7]

    def update_basis_functions(self):
        # Generate subintervals and basis functions for two-body
        for pair in self.interactions_map.get(2, []):
            if pair not in self.knots_map:  # compute knots if not provided
                r_min = self.r_min_map[pair]
                r_max = self.r_max_map[pair]
                n_intervals = self.resolution_map[pair]
                knot_sequence = self.knot_spacer(r_min, r_max, n_intervals)
                knot_sequence[knot_sequence == 0] = 1e-6
                self.knots_map[pair] = knot_sequence
            subintervals = get_knot_subintervals(self.knots_map[pair])
            self.knot_subintervals[pair] = subintervals
            self.basis_functions[pair] = generate_basis_functions(subintervals)
        # Generate subintervals and basis functions for two-body
        # Maps must contain three entries each.
        if self.degree > 2:
            for trio in self.interactions_map.get(3, []):
                if trio not in self.knots_map:
                    r_min = self.r_min_map[trio]
                    r_max = self.r_max_map[trio]
                    r_resolution = self.resolution_map[trio]
                    knot_sequences = []
                    for i in range(3):  # ij, ik, jk dimensions.
                        knot_sequence = self.knot_spacer(r_min[i],
                                                         r_max[i],
                                                         r_resolution[i])
                        knot_sequence[knot_sequence == 0] = 1e-6
                        knot_sequences.append(knot_sequence)
                    self.knots_map[trio] = knot_sequences
                subintervals = []
                basis_functions = []
                for knot_sequence in self.knots_map[trio]:
                    subinterval = get_knot_subintervals(knot_sequence)
                    basis_set = generate_basis_functions(subinterval)
                    subintervals.append(subinterval)
                    basis_functions.append(basis_set)
                self.knot_subintervals[trio] = subintervals
                self.basis_functions[trio] = basis_functions
            self.set_flatten_template_3B()
        self.partition_sizes = self.get_feature_partition_sizes()
        self.col_idx, self.frozen_c = self.generate_frozen_indices(
            offset_1b=self.offset_1b,
            n_trim=self.trailing_trim)

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

        TODO: refactor to break up into smaller, reusable functions

        Returns:
            combined_matrix (np.ndarray): regularization matrix made up of
                individual matrices per n-body interaction.
        """
        for k in kwargs:
            if k.lower()[0] == 'r':
                ridge_map[int(re.sub('[^0-9]', '', k))] = float(kwargs[k])
            elif k.lower()[0] == 'c':
                curvature_map[int(re.sub('[^0-9]', '', k))] = float(kwargs[k])
        ridge_map = {1: 1e-8, 2: 0.0, 3: 0.0, **ridge_map}
        curvature_map = {1: 0.0, 2: 1e-8, 3: 1e-8, **curvature_map}
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
                    matrix = regularize.get_penalty_matrix_3D(size[0] + 3,
                                                              size[1] + 3,
                                                              size[2] + 3,
                                                              ridge=r,
                                                              curvature=c)
                    mask = np.where(self.flat_weights[interaction] > 0)[0]
                    matrix = matrix[mask[None, :], mask[:, None]]
                else:
                    raise ValueError(
                        "Four-body terms and beyond are not yet implemented.")
                matrices.append(matrix)
        combined_matrix = regularize.combine_regularizer_matrices(matrices)
        return combined_matrix

    def get_feature_partition_sizes(self) -> List:
        """Get partition sizes: one-body, two-body, and three-body terms."""
        partition_sizes = [1] * len(self.chemical_system.element_list)
        for degree in range(2, self.chemical_system.degree + 1):
            interactions = self.chemical_system.interactions_map[degree]
            for interaction in interactions:
                if degree == 2:
                    size = self.resolution_map[interaction] + 3
                    partition_sizes.append(size)
                elif degree == 3:
                    mask = np.where(self.flat_weights[interaction] > 0)[0]
                    size = len(mask)
                    partition_sizes.append(size)
                else:
                    raise ValueError(
                        "Four-body terms and beyond are not yet implemented.")
        self.partition_sizes = partition_sizes
        return partition_sizes

    def get_interaction_partitions(self):
        interactions_list = self.interactions
        partition_sizes = self.get_feature_partition_sizes()
        offsets = np.cumsum(partition_sizes)
        offsets = np.insert(offsets, 0, 0)
        component_sizes = {}
        component_offsets = {}
        for j in range(len(interactions_list)):
            interaction = interactions_list[j]
            component_sizes[interaction] = partition_sizes[j]
            component_offsets[interaction] = offsets[j]
        return component_sizes, component_offsets

    def generate_frozen_indices(self,
                                offset_1b: bool = True,
                                n_trim: int = 3,
                                value: float = 0.0):
        pairs = self.interactions_map.get(2, [])
        trios = self.interactions_map.get(3, [])
        component_sizes, component_offsets = self.get_interaction_partitions()
        col_idx = []
        frozen_c = []
        for pair in pairs:
            offset = component_offsets[pair]
            size = component_sizes[pair]
            for trim_idx in range(1, n_trim + 1):
                idx = offset + size - trim_idx
                col_idx.append(idx)
                frozen_c.append(value)
        for trio in trios:
            template = np.zeros_like(self.templates[trio])
            for trim_idx in range(1, n_trim + 1):
                template[-trim_idx, :, :] = 1
                template[:, -trim_idx, :] = 1
                template[:, :, -trim_idx] = 1
            template = self.compress_3B(template, trio)
            mask = np.where(template > 0)[0]
            for idx in mask:
                col_idx.append(idx)
                frozen_c.append(value)
        if not offset_1b:
            for j in range(len(self.element_list)):
                col_idx.insert(0, j)
                frozen_c.insert(0, 0)
        col_idx = np.array(col_idx, dtype=int)
        frozen_c = np.array(frozen_c)
        return col_idx, frozen_c

    def set_flatten_template_3B(self):
        """
        Compute masks for flattening and unflattening 3B grid. The 3B BSpline
            set has three planes of symmetry corresponding to permutation
            of i, j, and k indices. Training is therefore performed with
            only the subset of basis functions corresponding to i < j < k.
            Basis functions on planes of symmetry have reduced weight.

        Returns:
            flat_weights (np.ndarray): vector of subset indices to use.
            unflatten_mask (np.ndarray): L x L x L boolean array for
                regenerating full basis function set.
        """
        if self.mask_trim:
            trailing_trim = self.trailing_trim
        else:
            trailing_trim = 0
        for trio in self.interactions_map[3]:
            l_space, m_space, n_space = self.knots_map[trio]
            template = angles.get_symmetry_weights(self.symmetry[trio],
                                                   l_space,
                                                   m_space,
                                                   n_space,
                                                   trailing_trim,)
            template_flat = template.flatten()
            template_mask, = np.where(template_flat > 0)
            self.template_mask[trio] = template_mask
            self.flat_weights[trio] = template_flat[template_mask]
            self.templates[trio] = template

    def compress_3B(self, grid, interaction):
        if self.symmetry[interaction] == 1:
            vec = grid.flatten()
        elif self.symmetry[interaction] == 2:
            vec = grid + grid.transpose(1, 0, 2)
            vec = vec.flat[self.template_mask[interaction]]
            vec = vec * self.flat_weights[interaction]
        elif self.symmetry[interaction] == 3:
            vec = (grid
                   + grid.transpose(0, 2, 1)
                   + grid.transpose(1, 0, 2)
                   + grid.transpose(1, 2, 0)
                   + grid.transpose(2, 0, 1)
                   + grid.transpose(2, 1, 0))
            vec = vec.flat[self.template_mask[interaction]]
            vec = vec * self.flat_weights[interaction]
        return vec

    def decompress_3B(self, vec, interaction):
        l_space, m_space, n_space = self.knots_map[interaction]
        L = len(l_space) - 4
        M = len(m_space) - 4
        N = len(n_space) - 4
        grid = np.zeros((L, M, N))
        grid.flat[self.template_mask[interaction]] = vec
        return grid


def get_knot_spacer(knot_strategy):
    # select knot spacing option
    if knot_strategy == 'lammps':
        spacing_function = generate_lammps_knots
    elif knot_strategy == 'linear':
        spacing_function = generate_uniform_knots
    elif knot_strategy == 'geometric':
        spacing_function = generate_geometric_knots
    elif knot_strategy == 'inverse':
        spacing_function = generate_inv_knots
    elif knot_strategy == 'custom':
        pass
    else:
        raise ValueError('Invalid value of knot_strategy:', knot_strategy)
    return spacing_function


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


def evaluate_basis_functions(points,
                             basis_functions,
                             nu=0,
                             trailing_trim=0,
                             flatten=True,
                             ):
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
    values_per_spline = [0] * n_splines
    for idx in range(n_splines - trailing_trim):
        # loop over number of basis functions
        bspline_values = basis_functions[idx](points, nu=nu)
        bspline_values[np.isnan(bspline_values)] = 0
        values_per_spline[idx] = bspline_values
    if not flatten:
        return values_per_spline
    value_per_spline = np.array([np.sum(values)
                                 for values in values_per_spline])
    return value_per_spline


def featurize_force_2B(basis_functions,
                       distances,
                       drij_dR,
                       knot_sequence,
                       trailing_trim=0,
                       ):
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
    for bspline_idx in np.arange(n_splines - trailing_trim):
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


def generate_inv_knots(r_min, r_max, n_intervals, sequence=True):
    """
    Generate knot points or knot sequence using an inverse transformation.
    This scheme yields higher resolution at smaller distances.

    Args:
        r_min (float): lower-bound for knot points.
        r_max (float): upper-bound for knot points.
        n_intervals (int): number of unique intervals in the knot sequence,
            i.e. n_intervals + 1 samples will be taken between r_min and r_max.
        sequence (bool): whether to repeat ends to yield knot sequence.

    Returns:
        knots (np.ndarray): knot points or knot sequence.
    """
    knots = np.linspace(1/r_min, 1/r_max, n_intervals + 1)**-1
    if sequence:
        knots = knot_sequence_from_points(knots)
    return knots


def generate_geometric_knots(r_min, r_max, n_intervals, sequence=True):
    """
    Generate knot points or knot sequence using a geometric progression.
    Points are evenly spaced on a log scale. This scheme yields higher
    resolution at smaller distances.

    Args:
        r_min (float): lower-bound for knot points.
        r_max (float): upper-bound for knot points.
        n_intervals (int): number of unique intervals in the knot sequence,
            i.e. n_intervals + 1 samples will be taken between r_min and r_max.
        sequence (bool): whether to repeat ends to yield knot sequence.

    Returns:
        knots (np.ndarray): knot points or knot sequence.
    """
    knots = np.geomspace(r_min, r_max, n_intervals + 1)
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
