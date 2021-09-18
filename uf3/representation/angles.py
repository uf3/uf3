import numpy as np
from numba import jit
import ase
from uf3.data import composition
from uf3.representation import bspline
from uf3.representation import distances
from scipy.spatial import distance


def featurize_energy_3b(geom: ase.Atoms,
                        knot_sets: List[List[np.ndarray]],
                        basis_functions: List,
                        hashes: List,
                        supercell: ase.Atoms = None,
                        ) -> List[np.ndarray]:
    """
    Args:
        geom (ase.Atoms)
        knot_sets (np.ndarray): list of lists of knot sequences per interaction
        basis_functions (list): list of lists of callable basis functions
            for each interaction
        hashes (list): list of three-body hashes.
        supercell (ase.Atoms)

    Returns:
        grid_3b (List of np.ndarray): feature grid per interaction.
    """
    if supercell is None:
        supercell = geom
    n_interactions = len(hashes)
    sup_comp = supercell.get_atomic_numbers()

    L, M, N = coefficient_counts_from_knots(knot_sets)
    # identify pairs
    dist_matrix, i_where, j_where = identify_ij(geom,
                                                knot_sets,
                                                supercell)

    grids = [np.zeros((L[i], M[i], N[i]))
             for i in range(n_interactions)]
    if len(i_where) == 0:
        return grids

    triplet_generator = generate_triplets(
        i_where, j_where, sup_comp, hashes, dist_matrix, knot_sets)

    for triplet_batch in triplet_generator:
        for interaction_idx in range(n_interactions):
            interaction_data = triplet_batch[interaction_idx]
            if interaction_data is None:
                continue
            i, r_l, r_m, r_n, ituples = interaction_data
            tuples_3b, idx_lmn = evaluate_triplet_distances(r_l, r_m, r_n,
                basis_functions[interaction_idx], knot_sets[interaction_idx])
            grids[interaction_idx] += arrange_3b(tuples_3b,
                                                 idx_lmn,
                                                 L[interaction_idx],
                                                 M[interaction_idx],
                                                 N[interaction_idx])
    return grids


def coefficient_counts_from_knots(knot_sets: List[List[np.ndarray]],
                                  ) -> Tuple[List[int], List[int], List[int]]:
    """
    Args:
        knot_sets (list): list of three-body knot sequences
            per chemical interation

    Returns:
        L, M, N: lists of number of basis functions per dimension (3).
    """
    L = []
    M = []
    N = []
    for knot_sequences in knot_sets:
        l_space, m_space, n_space = knot_sequences
        L.append(len(l_space) - 4)
        M.append(len(m_space) - 4)
        N.append(len(n_space) - 4)
    return L, M, N


@jit(nopython=True, nogil=True)
def arrange_3b(triangle_values, idx_lmn, L, M, N):
    """
    Args:
        triangle_values (np.ndarray): array of shape (n_triangles * 4, 3)
        idx_lmn (np.ndarray): array of shape (n_triangles * 4, 3)
        L (int): number of basis functions.

    Returns:
        grid (np.ndarray)
    """
    # multiply spline values and arrange in 3D grid
    grid = np.zeros((L, M, N))
    n_values = len(triangle_values)
    n_triangles = int(n_values / 4)
    for triangle_idx in range(n_triangles):
        idx_l = idx_lmn[triangle_idx * 4, 0]
        idx_m = idx_lmn[triangle_idx * 4, 1]
        idx_n = idx_lmn[triangle_idx * 4, 2]
        values = triangle_values[triangle_idx * 4: (triangle_idx + 1) * 4, :]
        # each triangle influences 4 x 4 x 4 = 64 basis functions
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    value = values[i, 0] * values[j, 1] * values[k, 2]
                    grid[idx_l + i, idx_m + j, idx_n + k] += value
    return grid


def featurize_force_3b(geom: ase.Atoms,
                       knot_sets: List[List[np.ndarray]],
                       basis_functions: List[List],
                       trio_hashes: Dict[int, np.ndarray],
                       supercell: ase.Atoms = None,
                       ) -> List[List[List[np.ndarray]]]:
    """
    Args:
        geom (ase.Atoms)
        knot_sets (np.ndarray): list of lists of knot sequences per interaction
        basis_functions (list): list of lists of callable basis functions
            for each chemical interaction
        trio_hashes (list)
        supercell (ase.Atoms)

    Returns:
        grid_3b (list): array-like list of shape
            (n_atoms, 3, n_basis_functions) where 3 refers to the three
            cartesian directions x, y, and z.
    """
    if supercell is None:
        supercell = geom
    n_interactions = len(trio_hashes)
    sup_comp = supercell.get_atomic_numbers()

    n_atoms = len(geom)
    L, M, N = coefficient_counts_from_knots(knot_sets)
    coords, matrix, x_where, y_where = identify_ij(geom,
                                                   knot_sets,
                                                   supercell,
                                                   square=True)
    if len(x_where) == 0:
        return np.zeros((n_atoms, 3, L, M, N))

    force_grids = [[[np.zeros((L[i], M[i], N[i])),
                     np.zeros((L[i], M[i], N[i])),
                     np.zeros((L[i], M[i], N[i]))]
                    for _ in range(n_atoms)]
                   for i in range(n_interactions)]
    # process each atom's neighbors to limit memory requirement
    triplet_generator = generate_triplets(
        x_where, y_where, sup_comp, trio_hashes, matrix, knot_sets)

    for triplet_batch in triplet_generator:
        for interaction_idx in range(n_interactions):
            interaction_data = triplet_batch[interaction_idx]
            if interaction_data is None:
                continue
            i, r_l, r_m, r_n, ituples = interaction_data

            tuples_3b, idx_lmn = evaluate_triplet_derivatives(r_l, r_m, r_n,
                basis_functions[interaction_idx], knot_sets[interaction_idx])

            drij_dr = distances.compute_direction_cosines(coords,
                                                          matrix,
                                                          ituples[:, 0],
                                                          ituples[:, 1],
                                                          n_atoms)
            drik_dr = distances.compute_direction_cosines(coords,
                                                          matrix,
                                                          ituples[:, 0],
                                                          ituples[:, 2],
                                                          n_atoms)
            drjk_dr = distances.compute_direction_cosines(coords,
                                                          matrix,
                                                          ituples[:, 1],
                                                          ituples[:, 2],
                                                          n_atoms)
            grids = arrange_deriv_3b(tuples_3b,
                                     idx_lmn,
                                     drij_dr,
                                     drik_dr,
                                     drjk_dr,
                                     L[interaction_idx],
                                     M[interaction_idx],
                                     N[interaction_idx])
            for a in range(n_atoms):
                for c in range(3):
                    force_grids[interaction_idx][a][c] -= grids[a][c]
    return force_grids


@jit(nopython=True, nogil=True)
def arrange_deriv_3b(triangle_values,
                     idx_lmn,
                     drij_dr,
                     drik_dr,
                     drjk_dr,
                     L,
                     M,
                     N):
    """
    Args:
        triangle_values (np.ndarray): array of shape (n_triangles * 4, 3)
        idx_lmn (np.ndarray): array of shape (n_triangles * 4, 3)
        dr{ij, ik, jk}_dr (np.ndarray): direction-cosine arrays of shape
            (n_atoms, 3, n_triangles).
        L (int): number of basis functions.

    Returns:
        force_grids (list): array-like list of shape
            (n_atoms, 3, n_basis_functions) where 3 refers to the three
            cartesian directions x, y, and z.
    """
    # multiply spline values and arrange in 3D grid
    n_atoms = len(drij_dr)
    n_values = len(triangle_values)
    n_triangles = int(n_values / 4)
    force_grids = [(np.zeros((L, M, N)),
                    np.zeros((L, M, N)),
                    np.zeros((L, M, N)))
                   for _ in range(n_atoms)]
    for a in range(n_atoms):  # atom index
        for c in range(3):  # cartesian directions
            for tri_idx in range(n_triangles):
                dij = drij_dr[a, c, tri_idx]
                dik = drik_dr[a, c, tri_idx]
                djk = drjk_dr[a, c, tri_idx]
                if dij == 0 and dik == 0 and djk == 0:
                    continue
                idx_l = idx_lmn[tri_idx * 4, 0]
                idx_m = idx_lmn[tri_idx * 4, 1]
                idx_n = idx_lmn[tri_idx * 4, 2]
                v = triangle_values[tri_idx * 4: (tri_idx + 1) * 4, :]
                # each triangle influences 4 x 4 x 4 = 64 basis functions
                for i in range(4):
                    for j in range(4):
                        for k in range(4):
                            val = (v[i, 3] * v[j, 1] * v[k, 2] * dij +
                                   v[i, 0] * v[j, 4] * v[k, 2] * dik +
                                   v[i, 0] * v[j, 1] * v[k, 5] * djk)
                            force_grids[a][c][idx_l+i, idx_m+j, idx_n+k] += val
    return force_grids


def identify_ij(geom: ase.Atoms,
                knot_sets: List[List[np.ndarray]],
                supercell: ase.Atoms = None,
                square: bool = False):
    """
    Args:
        geom (ase.Atoms)
        knot_sets (np.ndarray): list of lists of knot sequences per interaction
        supercell (ase.Atoms)
        square (bool): whether to return square distance matrix (True)
            or truncate to atoms in unit cell (False).

    TODO: refactor to break up into smaller, reusable functions

    Returns:
        dist_matrix (np.ndarray): rectangular matrix of shape
            (n_atoms, n_supercell) where n_supercell is the number of
            atoms within the cutoff distance of any in-unit-cell atom.
        {i, j}_where (np.ndarray): unsorted list of atom indices
            over which to loop to obtain valid pair distances.
    """
    if supercell is None:
        supercell = geom
    knots_flat = np.concatenate([sequence for set_ in knot_sets
                                 for sequence in set_])
    r_min = np.min(knots_flat)
    r_max = np.max(knots_flat)
    sup_positions = supercell.get_positions()
    geo_positions = geom.get_positions()
    # mask atoms that aren't close to any unit-cell atom
    dist_matrix = distance.cdist(geo_positions, sup_positions)
    cutoff_mask = dist_matrix < r_max  # ignore atoms without in-cell neighbors
    cutoff_mask = np.any(cutoff_mask, axis=0)
    sup_positions = sup_positions[cutoff_mask, :]  # reduced distance matrix
    # enforce distance cutoffs
    n_geo = len(geo_positions)
    dist_matrix = distance.cdist(sup_positions, sup_positions)
    if square is False:
        cut_matrix = dist_matrix[:n_geo, :]
        dist_mask = (cut_matrix >= r_min) & (cut_matrix <= r_max)
        i_where, j_where = np.where(dist_mask)
        return dist_matrix, i_where, j_where
    else:
        dist_mask = (dist_matrix >= r_min) & (dist_matrix <= r_max)
        i_where, j_where = np.where(dist_mask)
        return sup_positions, dist_matrix, i_where, j_where


def legacy_generate_triplets(i_where:np.ndarray,
                             j_where: np.ndarray,
                             distance_matrix: np.ndarray,
                             knot_sequences: List[np.ndarray],
                             ) -> Tuple:
    """
    Identify unique "i-j-j'" tuples by combining provided i-j pairs, then
    compute i-j, i-k, and j-k pair distances from i-j-k tuples,
        distance matrix, and knot sequence for cutoffs.

    Args:
        i_where (np.ndarray): sorted "i" indices
        j_where (np.ndarray): sorted "j" indices
        distance_matrix (np.ndarray)
        knot_sequences (list of np.ndarray)

    Returns:
        tuples_idx (np.ndarray): array of shape (n_triangles, 3)
    """
    # find unique values of i (sorted such that i < j)
    i_values, group_sizes = np.unique(i_where, return_counts=True)
    # group j by values of i
    i_groups = np.array_split(j_where, np.cumsum(group_sizes)[:-1])
    # generate j-k combinations
    for i in range(len(i_groups)):
        tuples = np.array(np.meshgrid(i_groups[i],
                                      i_groups[i])).T.reshape(-1, 2)
        tuples = np.insert(tuples, 0, i_values[i], axis=1)
        # atoms j and k are interchangable; filter
        comparison_mask = (tuples[:, 1] < tuples[:, 2])
        tuples = tuples[comparison_mask]
        # extract distance tuples
        r_l = distance_matrix[tuples[:, 0], tuples[:, 1]]
        r_m = distance_matrix[tuples[:, 0], tuples[:, 2]]
        r_n = distance_matrix[tuples[:, 1], tuples[:, 2]]
        # mask by longest distance
        l_mask = (r_l >= knot_sequences[0][0]) & (
                    r_l <= knot_sequences[0][-1])
        m_mask = (r_m >= knot_sequences[1][0]) & (
                    r_m <= knot_sequences[1][-1])
        n_mask = (r_n >= knot_sequences[2][0]) & (
                    r_n <= knot_sequences[2][-1])
        dist_mask = np.logical_and.reduce([l_mask, m_mask, n_mask])
        r_l = r_l[dist_mask]
        r_m = r_m[dist_mask]
        r_n = r_n[dist_mask]
        tuples = tuples[dist_mask]
        yield i, r_l, r_m, r_n, tuples


def generate_triplets(i_where: np.ndarray,
                      j_where: np.ndarray,
                      sup_composition: np.ndarray,
                      hashes: np.ndarray,
                      distance_matrix: np.ndarray,
                      knot_sets: List[List[np.ndarray]]
                      ) -> List[Tuple]:
    """
    Identify unique "i-j-j'" tuples by combining provided i-j pairs, then
    compute i-j, i-k, and j-k pair distances from i-j-k tuples,
        distance matrix, and knot sequence for cutoffs.

    TODO: refactor to break up into smaller, reusable functions

    Args:
        i_where (np.ndarray): sorted "i" indices
        j_where (np.ndarray): sorted "j" indices
        sup_composition (np.ndarray): composition given by atomic numbers.
        hashes (np.ndarray): array of unique integer hashes for interactions.
        distance_matrix (np.ndarray)
        knot_sets (np.ndarray): list of lists of knot sequences per interaction
        knot_sequences (list of np.ndarray)

    Returns:
        tuples_idx (np.ndarray): array of shape (n_triangles, 3)
    """
    n_hashes = len(hashes)
    # find unique values of i (sorted such that i < j)
    i_values, group_sizes = np.unique(i_where, return_counts=True)
    # group j by values of i
    i_groups = np.array_split(j_where, np.cumsum(group_sizes)[:-1])
    # generate j-k combinations
    for i in range(len(i_groups)):
        tuples = np.array(np.meshgrid(i_groups[i],
                                      i_groups[i])).T.reshape(-1, 2)
        tuples = np.insert(tuples, 0, i_values[i], axis=1)

        comp_tuples = sup_composition[tuples]
        comp_tuples[:, 1:] = np.sort(comp_tuples[:, 1:], axis=1)

        ijk_hash = composition.get_szudzik_hash(comp_tuples)


        grouped_triplets = [None] * n_hashes
        for j, hash_ in enumerate(hashes):
            ituples = tuples[ijk_hash == hash_]
            if len(ituples) == 0:
                grouped_triplets[j] = None
                continue
            # atoms j and k are interchangable; filter
            comparison_mask = (ituples[:, 1] < ituples[:, 2])
            ituples = ituples[comparison_mask]
            # extract distance tuples
            r_l = distance_matrix[ituples[:, 0], ituples[:, 1]]
            r_m = distance_matrix[ituples[:, 0], ituples[:, 2]]
            r_n = distance_matrix[ituples[:, 1], ituples[:, 2]]
            # mask by longest distance
            l_mask = ((r_l >= knot_sets[j][0][0])
                      & (r_l <= knot_sets[j][0][-1]))
            m_mask = ((r_m >= knot_sets[j][1][0])
                      & (r_m <= knot_sets[j][1][-1]))
            n_mask = ((r_n >= knot_sets[j][2][0])
                      & (r_n <= knot_sets[j][2][-1]))
            dist_mask = np.logical_and.reduce([l_mask, m_mask, n_mask])
            r_l = r_l[dist_mask]
            r_m = r_m[dist_mask]
            r_n = r_n[dist_mask]
            ituples = ituples[dist_mask]
            grouped_triplets[j] = i, r_l, r_m, r_n, ituples
        yield grouped_triplets


def evaluate_triplet_distances(r_l: np.ndarray,
                               r_m: np.ndarray,
                               r_n: np.ndarray,
                               basis_functions: List[List],
                               knot_sequences: List[np.ndarray],
                               trailing_trim: int = 3):
    """
    Identify non-zero basis functions for each point and call functions.

    TODO: refactor to break up into smaller, reusable functions

    Args:
        r_l (np.ndarray): vector of i-j distances.
        r_m (np.ndarray): vector of i-k distances.
        r_n (np.ndarray): vector of j-k distances.
        basis_functions (list): list of lists of of callable basis functions.
        knot_sequences (list of np.ndarray)
        trailing_trim (int): number of basis functions at trailing edge
            to suppress. Useful for ensuring smooth cutoffs.

    Returns:
        tuples_3b (np.ndarray):
        idx_lmn (np.ndarray):
    """
    l_knots, m_knots, n_knots = knot_sequences
    # identify non-zero splines (tiling each distance x 4)
    r_l, idx_rl = bspline.find_spline_indices(r_l, l_knots)
    r_m, idx_rm = bspline.find_spline_indices(r_m, m_knots)
    r_n, idx_rn = bspline.find_spline_indices(r_n, n_knots)
    # evaluate splines per dimension
    L = len(l_knots) - 4 - trailing_trim
    M = len(m_knots) - 4 - trailing_trim
    N = len(n_knots) - 4 - trailing_trim
    n_tuples = len(r_l)
    values_3b = np.zeros((n_tuples, 3))  # array of basis-function values
    for l_idx in range(L):
        mask = (idx_rl == l_idx)
        points = r_l[mask]
        values_3b[mask, 0] = basis_functions[0][l_idx](points)
    for m_idx in range(M):
        mask = (idx_rm == m_idx)
        points = r_m[mask]
        values_3b[mask, 1] = basis_functions[1][m_idx](points)
    for n_idx in range(N):
        mask = (idx_rn == n_idx)
        points = r_n[mask]
        values_3b[mask, 2] = basis_functions[2][n_idx](points)
    idx_lmn = np.vstack([idx_rl, idx_rm, idx_rn]).T
    return values_3b, idx_lmn


def evaluate_triplet_derivatives(r_l: np.ndarray,
                                 r_m: np.ndarray,
                                 r_n: np.ndarray,
                                 basis_functions: List[List],
                                 knot_sequences: List[np.ndarray],
                                 trailing_trim: int = 3
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify non-zero basis functions for each point and call functions.

    TODO: refactor to break up into smaller, reusable functions

    Args:
        r_l (np.ndarray): vector of i-j distances.
        r_m (np.ndarray): vector of i-k distances.
        r_n (np.ndarray): vector of j-k distances.
        basis_functions (list): list of lists of of callable basis functions.
        knot_sequences (list of np.ndarray)
        trailing_trim (int): number of basis functions at trailing edge
            to suppress. Useful for ensuring smooth cutoffs.

    Returns:
        tuples_3b (np.ndarray): tuples of spline derivative values
            per dimension per triangle.
        idx_lmn (np.ndarray): corresponding indices of relevant basis functions
            to increment with derivative values.
    """
    l_knots, m_knots, n_knots = knot_sequences
    # identify non-zero splines (tiling each distance x 4)
    r_l, idx_rl = bspline.find_spline_indices(r_l, l_knots)
    r_m, idx_rm = bspline.find_spline_indices(r_m, m_knots)
    r_n, idx_rn = bspline.find_spline_indices(r_n, n_knots)
    # evaluate splines per dimension
    L = len(l_knots) - 4 - trailing_trim
    M = len(m_knots) - 4 - trailing_trim
    N = len(n_knots) - 4 - trailing_trim
    n_tuples = len(r_l)
    values_3b = np.zeros((n_tuples, 6))  # array of basis-function values
    for l_idx in range(L):
        mask = (idx_rl == l_idx)
        points = r_l[mask]
        values_3b[mask, 0] = basis_functions[0][l_idx](points)
        values_3b[mask, 3] = basis_functions[0][l_idx](points, nu=1)
    for m_idx in range(M):
        mask = (idx_rm == m_idx)
        points = r_m[mask]
        values_3b[mask, 1] = basis_functions[1][m_idx](points)
        values_3b[mask, 4] = basis_functions[1][m_idx](points, nu=1)
    for n_idx in range(N):
        mask = (idx_rn == n_idx)
        points = r_n[mask]
        values_3b[mask, 2] = basis_functions[2][n_idx](points)
        values_3b[mask, 5] = basis_functions[2][n_idx](points, nu=1)
    idx_lmn = np.vstack([idx_rl, idx_rm, idx_rn]).T
    return values_3b, idx_lmn


def unflatten_3B(coefficients, trio, bspline_config):
    """
    Unflatten 3b coefficient vector into 3D array.

    Args:
        coefficients (np.ndarray): vector of flattened coefficients.
        trio (tuple): 3b interaction tuple.
        bspline_config (bspline.BSplineBasis)

    Returns:
        unflattened (np.ndarray): 3D array.
    """
    mask = bspline_config.unflatten_mask[trio]
    unflattened = np.zeros_like(mask, dtype=float)
    unflattened[mask] = coefficients
    unflattened = symmetrize_3B(unflattened)
    return unflattened


def symmetrize_3B(grid_3b, symmetry=2):
    """
        Symmetrize 3D array with mirror planes, enforcing permutational
            invariance with respect to i, j , and k indices.
            This allows us to avoid sorting, which is slow.
    """
    template = np.ones_like(grid_3b)
    for i, j, k in np.ndindex(*template.shape):
        if symmetry == 2:
            if (i == j):
                template[i, j, k] = 0.5
        elif symmetry == 3:
            if (i == j and i == k):
                template[i, j, k] = 1 / 6
            elif (i == k) or (i == j) or (j == k):
                template[i, j, k] = 0.5
    grid_3b = grid_3b * template
    if symmetry == 2:
        images = [grid_3b, grid_3b.transpose(1, 0, 2)]
    elif symmetry == 3:
        images = [grid_3b,
                  grid_3b.transpose(0, 2, 1),
                  grid_3b.transpose(1, 0, 2),
                  grid_3b.transpose(1, 2, 0),
                  grid_3b.transpose(2, 0, 1),
                  grid_3b.transpose(2, 1, 0)]
    else:
        images = [grid_3b]
    grid_3b = np.sum(images, axis=0)
    return grid_3b


def get_symmetry_weights(symmetry, l_space, m_space, n_space):
    L = len(l_space) - 4
    M = len(m_space) - 4
    N = len(n_space) - 4

    template = np.ones((L, M, N))
    if symmetry == 2:  # one mirror plane (i and j interchangeable)
        for i, j, k in np.ndindex(*template.shape):
            if (i > j):
                template[i, j, k] = 0
            elif (i == j):
                template[i, j, k] = 0.5
    elif symmetry == 3:  # three mirror planes (i, j, k intercheangable)
        for i, j, k in np.ndindex(*template.shape):
            if i == j and i == k:
                template[i, j, k] = 1 / 6
            elif i > j or j > k:
                template[i, j, k] = 0
            elif (i == k) or (i == j) or (j == k):
                template[i, j, k] = 0.5
    # triangle distance restriction
    for i, j, k in np.ndindex(*template.shape):
        if l_space[i + 4] + m_space[j + 4] <= n_space[k]:
            template[i, j, k] = 0
        elif l_space[i + 4] + n_space[k + 4] <= m_space[j]:
            template[i, j, k] = 0
        elif m_space[j + 4] + n_space[k + 4] <= l_space[i]:
            template[i, j, k] = 0
    return template
