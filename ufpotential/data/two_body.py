import numpy as np
from scipy import spatial

import ase

def distances_by_interaction(geom,
                             pair_tuples,
                             r_min_map,
                             r_max_map,
                             supercell=None,
                             atomic=False):
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
        r_min = r_min_map[pair]
        r_max = r_max_map[pair]
        pair_numbers = ase.symbols.symbols2numbers(pair)
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


def derivatives_by_interaction(geom, supercell, pair_tuples,
                               r_min_map, r_max_map):
    """
    Identify pair distances within a supercell and derivatives for evaluating
    forces, subject to lower and upper bounds given by r_min_map
    and r_max_map, per pair interaction e.g. A-A, A-B, B-B, etc.

    Args:
        geom (ase.Atoms): unit cell ase.Atoms.
        pair_tuples (list): list of pair interactions.
            e.g. [(A-A), (A-B), (A-C), (B-B), ...)]
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
    r_max = np.max(list(r_max_map.values()))
    supercell = mask_supercell_with_radius(geom, supercell, r_max)
    distance_matrix = get_distance_matrix(supercell, supercell)
    sup_positions = supercell.get_positions()
    sup_composition = np.array(supercell.get_atomic_numbers())
    # loop through interactions
    distance_map = {}
    derivative_map = {}
    for pair in pair_tuples:
        pair_numbers = ase.symbols.symbols2numbers(pair)
        r_min = r_min_map[pair]
        r_max = r_max_map[pair]
        comp_mask = mask_matrix_by_pair_interaction(pair_numbers,
                                                    sup_composition,
                                                    sup_composition)
        cut_mask = (distance_matrix > r_min) & (distance_matrix < r_max)
        # valid distances across configuration
        interaction_mask = comp_mask & cut_mask
        distance_map[pair] = distance_matrix[interaction_mask]
        # corresponding derivatives, flattened
        x_where, y_where = np.where(interaction_mask)
        drij_dr = compute_drij_dR(sup_positions, distance_matrix,
                                  x_where, y_where, n_atoms)
        derivative_map[pair] = drij_dr
    return distance_map, derivative_map


def mask_supercell_with_radius(geom, supercell, r_max):
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


def mask_matrix_by_pair_interaction(pair,
                                    geo_composition,
                                    sup_composition=None):
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


def get_distance_matrix(geom, supercell=None):
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


def get_distance_derivatives(geom, supercell, r_min=0, r_max=10):
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

    cut_mask = (distance_matrix >= r_min) & (distance_matrix <= r_max)

    i_where, j_where = np.where(cut_mask)

    drij_dr = compute_drij_dR(sup_positions, distance_matrix,
                              i_where, j_where, n_atoms)
    distances = distance_matrix[cut_mask]
    return distances, drij_dr


def distances_from_geometry(geom, supercell=None, r_min=0, r_max=10):
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


def compute_drij_dR(sup_positions,
                    distance_matrix,
                    i_where,
                    j_where,
                    n_atoms):
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
    m_range = np.arange(n_atoms)
    im = m_range[:, None] == i_where[None, :]
    jm = m_range[:, None] == j_where[None, :]
    kronecker = jm.astype(int) - im.astype(int)  # n_atoms x n_distances

    delta_r = sup_positions[j_where, :] - sup_positions[i_where, :]
    # n_distances x 3
    rij = distance_matrix[i_where, j_where]  # n_distances

    drij_dr = (kronecker[:, None, :]
               * delta_r.T[None, :, :]
               / rij[None, None, :])
    return drij_dr
