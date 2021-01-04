import numpy as np
from scipy import spatial


def distances_by_interaction(geometry,
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
        geometry: ase.Atoms of interest.
        pair_tuples (list): list of interactions per degree
            e.g. [(A-A), (A-B), (A-C), (B-B), ...)]
        r_min_map (dict): map of minimum pair distance per interaction.
            e.g. {(A-A): 2.0, (A-B): 3.0, (B-B): 4.0}
        r_max_map (dict): map of maximum pair distance per interaction.
        supercell (optional): ase.Atoms output of get_supercell
            used to account for atoms in periodic images.
        atomic (bool): whether to split array into lists of vectors
            corresponding to each atom's atomic environment.

    Returns:
        distances_map (dict): for each interaction key (A-A, A-B, ...),
            flattened np.ndarray of pair distances within range
            or list of flattened np.ndarray if atomic is True.
    """
    distance_matrix = get_distance_matrix(geometry, supercell)
    if supercell is None:
        supercell = geometry
    geo_composition = np.array(geometry.get_chemical_symbols())
    sup_composition = np.array(supercell.get_chemical_symbols())
    s_geo = len(geometry)
    # loop through interactions
    if atomic:
        distances_map = {tuple_: [] for tuple_ in pair_tuples}
    else:
        distances_map = {}
    for pair in pair_tuples:
        r_min = r_min_map[pair]
        r_max = r_max_map[pair]
        comp_mask = mask_matrix_by_pair_interaction(pair,
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


def derivatives_by_interaction(geometry, supercell, pair_tuples,
                               r_min_map, r_max_map):
    """
    Identify pair distances within a supercell and derivatives for evaluating
    forces, subject to lower and upper bounds given by r_min_map
    and r_max_map, per pair interaction e.g. A-A, A-B, B-B, etc.

    Args:
        geometry: unit cell ase.Atoms.
        pair_tuples (list): list of interactions per degree
            e.g. [(A-A), (A-B), (A-C), (B-B), ...)]
        r_min_map (dict): map of minimum pair distance per interaction.
            e.g. {(A-A): 2.0, (A-B): 3.0, (B-B): 4.0}
        r_max_map (dict): map of maximum pair distance per interaction.
        supercell (optional): ase.Atoms output of get_supercell
            used to account for atoms in periodic images.

    Returns:
        distances: flattened np.ndarray of pair distances across supercell
            within range.
        drij_dR: np.ndarray of shape (n_geo, 3, n_distances)
            and the second dimension corresponds to x, y, and z directions.
            Used to evaluate forces.
    """
    if supercell is None:
        supercell = geometry
    n_geo = len(geometry)
    # extract atoms from supercell that are within the maximum
    # cutoff distance of atoms in the unit cell.
    r_max = np.max(list(r_max_map.values()))
    supercell = mask_supercell_with_radius(geometry, supercell, r_max)
    distance_matrix = get_distance_matrix(supercell, supercell)
    sup_positions = supercell.get_positions()
    sup_composition = np.array(supercell.get_chemical_symbols())
    # loop through interactions
    distance_map = {}
    derivative_map = {}
    for pair in pair_tuples:
        r_min = r_min_map[pair]
        r_max = r_max_map[pair]
        comp_mask = mask_matrix_by_pair_interaction(pair,
                                                    sup_composition,
                                                    sup_composition)
        cut_mask = (distance_matrix > r_min) & (distance_matrix < r_max)
        # valid distances across configuration
        interaction_mask = comp_mask & cut_mask
        distance_map[pair] = distance_matrix[interaction_mask]
        # corresponding derivatives, flattened
        x_where, y_where = np.where(interaction_mask)
        n_distances = len(x_where)
        drij_dR = compute_drij_dR(sup_positions, distance_matrix,
                                  x_where, y_where, n_geo, n_distances)
        derivative_map[pair] = drij_dR
    return distance_map, derivative_map


def mask_supercell_with_radius(geometry, supercell, r_max):
    """
    Makes a copy of supercell and deletes atoms that are further than
    r_max away from any atom in the unit cell geometry.

    Args:
        geometry (ase.Atoms): unit cell of interest
        supercell (ase.Atoms): supercell, centered on unitcell.
        r_max (float): maximum radius.

    Returns:
        supercell (ase.Atoms): copy of supercell with subset of atoms
            within distance.
    """
    supercell = supercell.copy()
    distance_matrix = get_distance_matrix(geometry, supercell)
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
        vectors i.e. from ase.Atoms.get_chemical_symbols()

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
        comp_mask[geo_match[None, :], sup_match[:, None]] = 1
    return comp_mask


def get_distance_matrix(geometry, supercell=None):
    """
    Get distance matrix from geometry and, optionally,
    supercell including atoms from adjacent images.

    Args:
        geometry: ase.Atoms of interest.
        supercell (optional): ase.Atoms output of get_supercell
            used to account for atoms in periodic images.

    Returns:
        distance_matrix (np.ndarray): square (n x n) if supercell
            is not provided, rectangular (n x m) otherwise, where
            n and m are the number of atoms in the geometry
            and its supercell, respectively.
    """
    if supercell is None:
        supercell = geometry
    sup_positions = supercell.get_positions()
    geo_positions = geometry.get_positions()
    distance_matrix = spatial.distance.cdist(geo_positions, sup_positions)
    return distance_matrix


def get_distance_derivatives(geometry, supercell, r_min=0, r_max=10):
    """
    Identify pair distances within a supercell, subject to r_min < r < r_max,
    along with derivatives for evaluating forces. Legacy function
    for unary systems.

    Args:
        supercell: ase.Atoms output of get_supercell
            used to account for atoms in periodic images.
        geometry: unit cell ase.Atoms.
        r_min: minimum pair distance to consider.
        r_max: maximum pair distance to consider.

    Returns:
        distances: flattened np.ndarray of pair distances across supercell
            within range.
        drij_dR: np.ndarray of shape (n_geo, 3, n_distances)
            and the second dimension corresponds to x, y, and z directions.
            Used to evaluate forces.
    """
    sup_positions = supercell.get_positions()
    geo_positions = geometry.get_positions()
    distance_matrix = spatial.distance.cdist(geo_positions, sup_positions)
    weight_mask = distance_matrix <= r_max  # mask distance matrix by r_max
    valids_mask = np.any(weight_mask, axis=0)
    sup_positions = sup_positions[valids_mask, :]

    distance_matrix = spatial.distance.cdist(sup_positions, sup_positions)
    n_geo = len(geo_positions)

    cut_mask = (distance_matrix >= r_min) & (distance_matrix <= r_max)

    x_where, y_where = np.where(cut_mask)
    n_distances = len(x_where)

    drij_dR = compute_drij_dR(sup_positions, distance_matrix,
                              x_where, y_where, n_geo, n_distances)
    distances = distance_matrix[cut_mask]
    return distances, drij_dR


def distances_from_geometry(geometry, supercell=None, r_min=0, r_max=10):
    """
    Identify pair distances within a geometry (or between a geometry and its
    supercell), subject to r_min < r < r_max. Legacy function
    for unary systems.

    Args:
        geometry: ase.Atoms of interest.
        supercell (optional): ase.Atoms output of get_supercell
            used to account for atoms in periodic images.
        r_min (optional): minimum pair distance; default = 0.
        r_max (optional): maximum pair distance; default  = 10.

    Returns:
        flattened np.ndarray of pair distances within range.
    """
    distance_matrix = get_distance_matrix(geometry, supercell)
    cut_mask = (distance_matrix > r_min) & (distance_matrix < r_max)
    distances = distance_matrix[cut_mask]
    return distances


def compute_drij_dR(sup_positions, distance_matrix,
                    i_where, j_where,
                    n_geo, n_distances):
    """
    Pure-python port of intermediate function for computing derivatives
    for forces. Cython implementation available upon request.
    TODO: acceleration with Numba.

    Args:
        sup_positions: atom positions in supercell.
        distance_matrix: output of spatial.distance.cdist(sup_positions,
            sup_positions).
        i_where: indices of i-th atom based on distance mask.
        j_where: indices of j-th atom based on distance mask.
        n_geo: number of atoms in original entry.
        n_distances: number of pair distances of interest.

    Returns:
        drij_dR: np.ndarray of shape (n_geo, 3, n_distances)
            and the second dimension corresponds to x, y, and z directions.
            Used to evaluate forces.
    """
    drij_dR = np.zeros((n_geo, 3, n_distances))

    for idx in range(n_distances):
        i = i_where[idx]  # loop over only i and j within distance constraints
        j = j_where[idx]

        for c_idx in range(3):
            Rij = sup_positions[j, c_idx] - sup_positions[i, c_idx]
            for m in range(n_geo):  # loop over m index
                if j == m:
                    jm = 1
                else:
                    jm = 0
                if i == m:
                    im = 1
                else:
                    im = 0
                delta = jm - im
                drij_dR[m, c_idx, idx] = delta * Rij / distance_matrix[i, j]
    return drij_dR
