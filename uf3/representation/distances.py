"""
This module provides functions for computing neighbor lists, evaluating
pair distances, computing direction cosines for force components, and
fitting/evaluating one-dimensional BSplines.
"""

from typing import List, Dict, Tuple, Union, Any
import numpy as np
import numba as nb
from scipy import spatial
from scipy import signal
import ase
from ase import symbols as ase_symbols
from uf3.data import geometry
from uf3.data import composition
from uf3.util import parallel


def distances_by_interaction(geom: ase.Atoms,
                             pair_tuples: List[Tuple[str]],
                             r_min_map: Dict[Tuple[str], float],
                             r_max_map: Dict[Tuple[str], float],
                             supercell: ase.Atoms = None,
                             atomic: bool = False,
                             ) -> Dict[Tuple[str], np.ndarray]:
    """
    Identify pair distances within an entry (or between an entry and its
    supercell), subject to lower and upper bounds given by r_min_map
    and r_max_map, per pair interaction e.g. A-A, A-B, B-B, etc.

    Args:
        geom (ase.Atoms): configuration of interest.
        pair_tuples (list): list of pair interactions
            e.g. [(A-A), (A-B), (A-C), (B-B), ...)]
        r_min_map (dict): map of minimum pair distance per interaction
            e.g. {(A-A): 2.0, (A-B): 3.0, (B-B): 4.0}
        r_max_map (dict): map of maximum pair distance per interaction
        supercell (ase.Atoms): ase.Atoms output of get_supercell
            used to account for atoms in periodic images.
        atomic (bool): whether to split array into lists of vectors
            corresponding to each atom's atomic environment.

    Returns:
        distances_map (dict): for each interaction key (A-A, A-B, ...),
            flattened np.ndarray of pair distances within range
            or list of flattened np.ndarray if atomic is True.
    """
    distance_matrix = get_distance_matrix(geom, supercell)
    if supercell is None:
        supercell = geom
    geo_composition = np.array(geom.get_atomic_numbers())
    sup_composition = np.array(supercell.get_atomic_numbers())
    s_geo = len(geom)
    # loop through interactions
    if atomic:
        distances_map = {tuple_: [] for tuple_ in pair_tuples}
    else:
        distances_map = {}
    for pair in pair_tuples:
        r_min = max(r_min_map[pair], 0)
        r_max = r_max_map[pair]
        pair_numbers = ase_symbols.symbols2numbers(pair)
        comp_mask = mask_matrix_by_pair_interaction(pair_numbers,
                                                    geo_composition,
                                                    sup_composition)
        cut_mask = (distance_matrix > r_min) & (distance_matrix < r_max)
        if not atomic:  # valid distances across configuration
            interaction_mask = comp_mask & cut_mask
            distances_map[pair] = distance_matrix[interaction_mask]
        else:  # valid distances per atom
            for i in range(s_geo):
                atom_slice = distance_matrix[i]
                interaction_mask = comp_mask[i] & cut_mask[i]
                distances_map[pair].append(atom_slice[interaction_mask])
    return distances_map


def derivatives_by_interaction(geom: ase.Atoms,
                               pair_tuples: List[Tuple[str]],
                               r_cut: float,
                               r_min_map: Dict[Tuple[str], float],
                               r_max_map: Dict[Tuple[str], float],
                               supercell: ase.Atoms = None,
                               ) -> Tuple[Dict, Dict]:
    """
    Identify pair distances within a supercell and derivatives for evaluating
    forces, subject to lower and upper bounds given by r_min_map
    and r_max_map, per pair interaction e.g. A-A, A-B, B-B, etc.

    Args:
        geom (ase.Atoms): unit cell ase.Atoms.
        pair_tuples (list): list of pair interactions.
            e.g. [(A-A), (A-B), (A-C), (B-B), ...)]
        r_cut (float): cutoff radius (angstroms).
        r_min_map (dict): map of minimum pair distance per interaction.
            e.g. {(A-A): 2.0, (A-B): 3.0, (B-B): 4.0}
        r_max_map (dict): map of maximum pair distance per interaction.
        supercell (ase.Atoms): ase.Atoms output of get_supercell
            used to account for atoms in periodic images.

    Returns:
        distances: flattened np.ndarray of pair distances across supercell
            within range.
        drij_dR: np.ndarray of shape (n_atoms, 3, n_distances)
            and the second dimension corresponds to x, y, and z directions.
            Used to evaluate forces.
    """
    if supercell is None:
        supercell = geom
    n_atoms = len(geom)
    # extract atoms from supercell that are within the maximum
    # cutoff distance of atoms in the unit cell.
    supercell = mask_supercell_with_radius(geom, supercell, r_cut)
    distance_matrix = get_distance_matrix(supercell, supercell)

    # the bottom right block where both i and j are ghost atoms are unnecessary
    n_supercell = len(supercell)
    real_row_idx = np.arange(n_supercell).reshape(n_supercell, 1)
    real_col_idx = np.arange(n_supercell).reshape(1, n_supercell)
    real_mask = (real_row_idx < n_atoms) | (real_col_idx < n_atoms)

    sup_positions = supercell.get_positions()
    sup_composition = np.array(supercell.get_atomic_numbers())
    # loop through interactions
    distance_map = {}
    derivative_map = {}
    for pair in pair_tuples:
        pair_numbers = ase_symbols.symbols2numbers(pair)
        r_min = max(r_min_map[pair], 0)
        r_max = r_max_map[pair]
        comp_mask = mask_matrix_by_pair_interaction(pair_numbers,
                                                    sup_composition,
                                                    sup_composition)
        cut_mask = (distance_matrix > r_min) & (distance_matrix < r_max)
        # valid distances across configuration
        interaction_mask = real_mask & comp_mask & cut_mask
        distance_map[pair] = distance_matrix[interaction_mask]
        # corresponding derivatives, flattened
        x_where, y_where = np.where(interaction_mask)
        drij_dr = compute_direction_cosines(sup_positions, distance_matrix,
                                            x_where, y_where, n_atoms)
        derivative_map[pair] = drij_dr
    return distance_map, derivative_map


def mask_supercell_with_radius(geom: ase.Atoms,
                               supercell: ase.Atoms,
                               r_max: float,
                               ) -> ase.Atoms:
    """
    Makes a copy of supercell and deletes atoms that are further than
    r_max away from any atom in the unit cell geometry.

    Args:
        geom (ase.Atoms): unit cell of interest
        supercell (ase.Atoms): supercell, centered on unitcell.
        r_max (float): maximum radius.

    Returns:
        supercell (ase.Atoms): copy of supercell with subset of atoms
            within distance.
    """
    supercell = supercell.copy()
    distance_matrix = get_distance_matrix(geom, supercell)
    weight_mask = distance_matrix <= r_max  # mask distance matrix by r_max
    valids_mask = np.any(weight_mask, axis=0)
    reject_idx = np.where(valids_mask == False)[0]
    del supercell[reject_idx]
    return supercell


def mask_matrix_by_pair_interaction(pair: Union[List, Tuple],
                                    geo_composition: np.ndarray,
                                    sup_composition: np.ndarray = None
                                    ) -> np.ndarray:
    """
    Generates boolean mask for the distance matrix based on composition
        vectors i.e. from ase.Atoms.get_atomic_numbers()

        e.g. identifying A-B interactions:
                              supercell
                        A   B   A   A   B   B
                        ---------------------
        geometry    A | -   X   -   -   X   X
                    B | X   -   X   X   -   -

    Args:
        pair (tuple): pair interaction of interest e.g. (A-B)
        geo_composition (list, np.ndarray): list of elements
            for each atom in geometry.
        sup_composition (list, np.ndarray): optional list of elements
            for each atom in supercell.

    Returns:
        comp_mask (np.ndarray): Boolean mask of dimension (n x m)
            corresponding to pair interactions of the specified type,
            where n and m are the number of atoms in the geometry
            and its supercell, respectively.
    """
    if sup_composition is None:
        sup_composition = geo_composition
    s_geo = len(geo_composition)
    s_sup = len(sup_composition)
    comp_mask = np.zeros((s_geo, s_sup), dtype=bool)
    for i, j in (pair, pair[::-1]):
        geo_match = np.where(geo_composition == j)[0]
        sup_match = np.where(sup_composition == i)[0]
        comp_mask[geo_match[None, :], sup_match[:, None]] = True
    return comp_mask


def get_distance_matrix(geom: ase.Atoms,
                        supercell: ase.Atoms = None,
                        ) -> np.ndarray:
    """
    Get distance matrix from geometry and, optionally,
    supercell including atoms from adjacent images.

    Args:
        geom (ase.Atoms): unit cell of interest.
        supercell (ase.Atoms): ase.Atoms output of get_supercell
            used to account for atoms in periodic images.

    Returns:
        distance_matrix (np.ndarray): square (n x n) if supercell
            is not provided, rectangular (n x m) otherwise, where
            n and m are the number of atoms in the geometry
            and its supercell, respectively.
    """
    if supercell is None:
        supercell = geom
    sup_positions = supercell.get_positions()
    geo_positions = geom.get_positions()
    distance_matrix = spatial.distance.cdist(geo_positions, sup_positions)
    return distance_matrix


def get_distance_derivatives(geom: ase.Atoms,
                             supercell: ase.Atoms,
                             r_min: float = 0.0,
                             r_max: float = 10.0,
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify pair distances within a supercell, subject to r_min < r < r_max,
    along with derivatives for evaluating forces. Legacy function
    for unary systems.

    Args:
        supercell (ase.Atoms): output of get_supercell
            used to account for atoms in periodic images.
        geom (ase.Atoms): unit cell ase.Atoms.
        r_min (float): minimum pair distance to consider.
        r_max (float): maximum pair distance to consider.

    Returns:
        distances: flattened np.ndarray of pair distances across supercell
            within range.
        drij_dR: np.ndarray of shape (n_atoms, 3, n_distances)
            and the second dimension corresponds to x, y, and z directions.
            Used to evaluate forces.
    """
    sup_positions = supercell.get_positions()
    geo_positions = geom.get_positions()
    distance_matrix = spatial.distance.cdist(geo_positions, sup_positions)
    weight_mask = distance_matrix <= r_max  # mask distance matrix by r_max
    valids_mask = np.any(weight_mask, axis=0)
    sup_positions = sup_positions[valids_mask, :]

    distance_matrix = spatial.distance.cdist(sup_positions, sup_positions)
    n_atoms = len(geo_positions)
    r_min = max(r_min, 0)
    cut_mask = (distance_matrix > r_min) & (distance_matrix <= r_max)
    i_where, j_where = np.where(cut_mask)
    drij_dr = compute_direction_cosines(sup_positions, distance_matrix,
                                        i_where, j_where, n_atoms)
    distances = distance_matrix[cut_mask]
    return distances, drij_dr


def distances_from_geometry(geom: ase.Atoms,
                            supercell: ase.Atoms = None,
                            r_min: float = 0.0,
                            r_max: float = 10.0,
                            ) -> np.ndarray:
    """
    Identify pair distances within a geometry (or between a geometry and its
    supercell), subject to r_min < r < r_max. Legacy function
    for unary systems.

    Args:
        geom (ase.Atoms): unit cell of interest.
        supercell (ase.Atoms): output of get_supercell used to account for
            atoms in periodic images.
        r_min (float): minimum pair distance; default = 0.
        r_max (float): maximum pair distance; default  = 10.

    Returns:
        flattened np.ndarray of pair distances within range.
    """
    distance_matrix = get_distance_matrix(geom, supercell)
    cut_mask = (distance_matrix > r_min) & (distance_matrix < r_max)
    distances = distance_matrix[cut_mask]
    return distances


@nb.njit(nb.float64[:, :](nb.int32[:], nb.int64[:], nb.int64[:]))
def kronecker_delta(m_range, i_where, j_where):
    n_atoms = len(m_range)
    n_pairs = len(i_where)
    kronecker = np.zeros((n_atoms, n_pairs), dtype=nb.float64)

    for m in m_range:
        for idx in range(n_pairs):
            i = i_where[idx]
            j = j_where[idx]
            kronecker[m, idx] = (m == j) - (m == i)
    return kronecker


def kronecker_vectorized(n_atoms: int,
                         i_where: np.ndarray,
                         j_where: np.ndarray
                         ) -> np.ndarray:
    m_range = np.arange(n_atoms)
    im = m_range[:, None] == i_where[None, :]
    jm = m_range[:, None] == j_where[None, :]
    kronecker = jm.astype(int) - im.astype(int)
    return kronecker


def compute_direction_cosines(sup_positions: np.ndarray,
                              distance_matrix: np.ndarray,
                              i_where: np.ndarray,
                              j_where: np.ndarray,
                              n_atoms: int
                              ) -> np.ndarray:
    """
    Intermediate function for computing derivatives for forces.

    Args:
        sup_positions: atom positions in supercell.
        distance_matrix: output of spatial.distance.cdist(sup_positions,
            sup_positions).
        i_where: indices of i-th atom based on distance mask.
        j_where: indices of j-th atom based on distance mask.
        n_atoms: number of atoms in original unit cell.

    Returns:
        drij_dR: np.ndarray of shape (n_atoms, 3, n_distances)
            and the second dimension corresponds to x, y, and z directions.
            Used to evaluate forces.
    """
    # n_atoms x n_distances
    kronecker = kronecker_delta(np.arange(n_atoms, dtype=np.int32),
                                i_where,
                                j_where)
    # n_distances x 3
    delta_r = sup_positions[j_where, :] - sup_positions[i_where, :]
    # n_distances
    rij = distance_matrix[i_where, j_where]
    drij_dr = (kronecker[:, None, :]
               * delta_r.T[None, :, :]
               / rij[None, None, :])
    return drij_dr


def summarize_distances(geometries: List[ase.Atoms],
                        chemical_system: composition.ChemicalSystem,
                        r_cut: float = 12.0,
                        n_bins: int = 100,
                        print_stats: bool = True,
                        min_peak_width: float = 0.5,
                        progress: Any = "bar",
                        ) -> Tuple[Dict, np.ndarray, Dict]:
    """
    Construct histogram of distances per pair interaction across
        list of geometries. Useful for optimizing the lower- and upper-
        bounds of knot sequences.

    TODO: refactor to break up into smaller, reusable functions

    Args:
        geometries (list): list of ase.Atoms configuration.
        chemical_system (uf3.composition.ChemicalSystem)
        r_cut (float): cutoff distance in angstroms.
        n_bins (int): number of bins in histogram.
        print_stats (bool): print minimum distance and identified peaks.
        min_peak_width (float): minimum peak with in angstroms for
            peak-finding algorithm.
        progress: style of progress bar.

    Returns:
        histogram_map (dict): for each interaction key (A-A, A-B, ...),
            vector of histogram frequencies of length n_bins. Frequency
            values are divided by r^2, evaluated at the middle of each bin.
        bin_edges (np.ndarray): bin edges of histogram.
    """
    pair_tuples = chemical_system.interactions_map[2]
    bin_edges = np.linspace(0, r_cut, n_bins + 1)
    histogram_values = {pair: np.zeros(n_bins) for pair in pair_tuples}
    n_entries = len(geometries)
    iterable = parallel.progress_iter(geometries, style=progress)
    for geom in iterable:
        if any(geom.pbc):
            supercell = geometry.get_supercell(geom, r_cut=r_cut)
            density = len(geom) / geom.get_volume()
        else:
            supercell = geom
            density = 1
        distance_matrix = get_distance_matrix(geom, supercell)
        geo_composition = np.array(geom.get_atomic_numbers())
        sup_composition = np.array(supercell.get_atomic_numbers())
        # loop through interactions
        for pair in pair_tuples:
            pair_numbers = ase_symbols.symbols2numbers(pair)
            comp_mask = mask_matrix_by_pair_interaction(pair_numbers,
                                                        geo_composition,
                                                        sup_composition)
            cut_mask = (distance_matrix > 0) & (distance_matrix < r_cut)
            interaction_mask = comp_mask & cut_mask
            distances_vector = distance_matrix[interaction_mask]
            frequencies, _ = np.histogram(distances_vector, bin_edges)
            frequencies = frequencies / density / n_entries / 2
            if pair[0] != pair[1]:
                frequencies /= 2
            histogram_values[pair] += frequencies
    bin_centers = 0.5 * np.add(bin_edges[:-1], bin_edges[1:])
    bin_span = int(np.ceil(min_peak_width
                           / (bin_edges[1] - bin_edges[0])))
    lower_bounds = {}
    for pair in pair_tuples:
        histogram_values[pair] /= bin_centers ** 2 * 4 * np.pi
        lower_bound = bin_edges[np.nonzero(histogram_values[pair])[0][0]]
        lower_bounds[pair] = lower_bound
        if print_stats:
            peaks = bin_centers[signal.find_peaks(histogram_values[pair],
                                                  width=bin_span)[0]]
            print(pair, 'Lower bound: {0:.3f} angstroms'.format(lower_bound))
            print(pair,
                  'Peaks (min width {} angstroms):'.format(min_peak_width),
                  peaks)
    return histogram_values, bin_edges, lower_bounds
