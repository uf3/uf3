import numpy as np
from numba import jit
from uf3.representation import bspline
from uf3.representation import distances
from scipy.spatial import distance


# def mask_matrix_by_trio_interaction(tuples_idx,
#                                     interactions,
#                                     sup_composition):
#     comp_masks = {}
#     comp_tuples = np.zeros_like(tuples_idx, dtype=int)
#     comp_tuples[:, 0] = sup_composition[tuples_idx[:, 0]]
#     comp_tuples[:, 1] = sup_composition[tuples_idx[:, 1]]
#     comp_tuples[:, 2] = sup_composition[tuples_idx[:, 2]]
#     for interaction in interactions:
#         trio_numbers = ase.symbols.symbols2numbers(interaction)
#         mask = np.zeros(len(tuples_idx), dtype=bool)
#         for i, j, k in itertools.combinations(trio_numbers, 3):
#             i_match = (comp_tuples[:, 0] == i)
#             j_match = (comp_tuples[:, 1] == j)
#             k_match = (comp_tuples[:, 2] == k)
#             mask[i_match & j_match & k_match] = True
#         comp_masks[interaction] = mask
#     return comp_masks


def get_energy_3B(grid,
                  geom,
                  supercell,
                  knot_sequence,
                  basis_functions):
    """Evaluate energy contribution from 3B term."""
    r_min = np.min(knot_sequence)
    r_max = np.max(knot_sequence)
    ext_positions = supercell.get_positions()
    geo_positions = geom.get_positions()
    # mask atoms that aren't close to any unit-cell atom
    dist_matrix = distance.cdist(geo_positions, ext_positions)
    weight_mask = dist_matrix < r_max
    valids_mask = np.any(weight_mask, axis=0)
    ext_positions = ext_positions[valids_mask, :]
    # reduced distance matrix
    n_geo = len(geo_positions)
    ext_distances = distance.cdist(ext_positions, ext_positions)
    dist_matrix = ext_distances[:n_geo, :]
    cut_mask = (dist_matrix > r_min) & (dist_matrix < r_max)
    x_where, y_where = np.where(cut_mask)
    # TODO: extract method
    lower_where, upper_where = sort_pairs(x_where, y_where)
    tuples = generate_triplets(lower_where, upper_where)
    r_l, r_m, r_n, _ = get_triplet_distances(ext_distances,
                                             tuples,
                                             knot_sequence)
    # evaluate splines
    triangle_values, idx_lmn = evaluate_triplet_distances(
        r_l, r_m, r_n, basis_functions, knot_sequence)
    energy = evaluate_3b(triangle_values, idx_lmn, grid)
    return energy


def get_forces_3B(grid,
                  geom,
                  supercell,
                  knot_sequence,
                  basis_functions):
    """Evaluate force contributions from 3B term."""
    n_atoms = len(geom)
    r_min = np.min(knot_sequence)
    r_max = np.max(knot_sequence)
    sup_positions = supercell.get_positions()
    geo_positions = geom.get_positions()
    # mask atoms that aren't close to any unit-cell atom
    matrix = distance.cdist(geo_positions, sup_positions)
    weight_mask = matrix < r_max
    valids_mask = np.any(weight_mask, axis=0)
    coords = sup_positions[valids_mask, :]
    # reduced distance matrix
    matrix = distance.cdist(coords, coords)
    cut_mask = (matrix > r_min) & (matrix < r_max)
    x_where, y_where = np.where(cut_mask)
    # TODO: extract method
    lower_where, upper_where = sort_pairs(x_where, y_where)
    tuples = generate_triplets(lower_where, upper_where)
    r_l, r_m, r_n, mask = get_triplet_distances(matrix,
                                                tuples,
                                                knot_sequence)
    # evaluate splines
    triangle_values, idx_lmn = evaluate_triplet_derivs(
        r_l, r_m, r_n, basis_functions, knot_sequence)
    tuples = tuples[mask]
    tri_i = tuples[:, 0]
    tri_j = tuples[:, 1]
    tri_k = tuples[:, 2]
    drij_dr = distances.compute_direction_cosines(coords,
                                                  matrix,
                                                  tri_i,
                                                  tri_j,
                                                  n_atoms)
    drik_dr = distances.compute_direction_cosines(coords,
                                                  matrix,
                                                  tri_i,
                                                  tri_k,
                                                  n_atoms)
    drjk_dr = distances.compute_direction_cosines(coords,
                                                  matrix,
                                                  tri_j,
                                                  tri_k,
                                                  n_atoms)
    forces = evaluate_deriv_3b(triangle_values,
                               idx_lmn,
                               drij_dr,
                               drik_dr,
                               drjk_dr,
                               grid)
    return forces


@jit(nopython=True, nogil=True)
def evaluate_3b(triangle_values, idx_lmn, grid):
    """Evaluate energy contribution from 3B term."""
    # multiply spline values and arrange in 3D grid
    energy = 0.0
    n_values = len(triangle_values)
    n_triangles = int(n_values / 4)
    for triangle_idx in range(n_triangles):
        idx_l = idx_lmn[triangle_idx * 4, 0]
        idx_m = idx_lmn[triangle_idx * 4, 1]
        idx_n = idx_lmn[triangle_idx * 4, 2]
        values = triangle_values[triangle_idx * 4: (triangle_idx + 1) * 4, :]
        # for loops
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    spline = values[i, 0] * values[j, 1] * values[k, 2]
                    c = grid[idx_l + i, idx_m + j, idx_n + k]
                    energy += c * spline
    return energy


@jit(nopython=True, nogil=True)
def evaluate_deriv_3b(triangle_values,
                      idx_lmn,
                      drij_dr,
                      drik_dr,
                      drjk_dr,
                      grid):
    """Evaluate force contributions from 3B term."""
    # multiply spline values and arrange in 3D grid
    n_atoms = len(drij_dr)
    forces = np.zeros((n_atoms, 3))

    n_values = len(triangle_values)
    n_triangles = int(n_values / 4)
    for triangle_idx in range(n_triangles):
        idx_l = idx_lmn[triangle_idx * 4, 0]
        idx_m = idx_lmn[triangle_idx * 4, 1]
        idx_n = idx_lmn[triangle_idx * 4, 2]
        v = triangle_values[triangle_idx * 4: (triangle_idx + 1) * 4, :]
        # each triangle influences 4 x 4 x 4 = 64 basis functions
        for a in range(n_atoms):  # atom index
            for c in range(3):  # cartesian directions
                dij = drij_dr[a, c, triangle_idx]
                dik = drik_dr[a, c, triangle_idx]
                djk = drjk_dr[a, c, triangle_idx]
                for i in range(4):
                    for j in range(4):
                        for k in range(4):
                            val = (v[i, 3] * v[j, 1] * v[k, 2] * dij +
                                   v[i, 0] * v[j, 4] * v[k, 2] * dik +
                                   v[i, 0] * v[j, 1] * v[k, 5] * djk)
                            coef = grid[idx_l + i, idx_m + j, idx_n + k]
                            forces[a, c] += coef * val
    return forces


def featurize_energy_3B(geom, supercell, knot_sequence, basis_functions):
    """
    Args:
        geom (ase.Atoms)
        supercell (ase.Atoms)
        knot_sequence (np.ndarray)
        basis_functions (list): list of callable basis functions.

    Returns:
        grid_3b (np.ndarray)
    """
    # identify pairs
    dist_matrix, i_where, j_where = identify_ij(geom,
                                                knot_sequence,
                                                supercell)
    # enforce i < j
    i_where, j_where = sort_pairs(i_where, j_where)
    # generate valid i, j, k triplets by joining i-j and i-j' pairs
    tuples_ijk = generate_triplets(i_where, j_where)

    # query distance matrix for i-j, i-k, j-k distances
    r_l, r_m, r_n, _ = get_triplet_distances(dist_matrix,
                                             tuples_ijk,
                                             knot_sequence)
    # evaluate splines
    tuples_3b, idx_lmn = evaluate_triplet_distances(
        r_l, r_m, r_n, basis_functions, knot_sequence)
    # arrange spline values into grid - BOTTLENECK
    L = len(knot_sequence) - 4
    grid_3b = arrange_3B(tuples_3b, idx_lmn, L)
    # enforce symmetry
    grid_3b = symmetrize_3B(grid_3b)
    return grid_3b


@jit(nopython=True, nogil=True)
def arrange_3B(triangle_values, idx_lmn, L):
    """
    Args:
        triangle_values (np.ndarray): array of shape (n_triangles * 4, 3)
        idx_lmn (np.ndarray): array of shape (n_triangles * 4, 3)
        L (int): number of basis functions.

    Returns:
        grid (np.ndarray)
    """
    # multiply spline values and arrange in 3D grid
    M = L
    N = L
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


def featurize_force_3B(geom, sup_geometry, knot_sequence, basis_functions,
                       batch_size=10000):
    """
    Args:
        geom (ase.Atoms)
        sup_geometry (ase.Atoms)
        knot_sequence (np.ndarray)
        basis_functions (list): list of callable basis functions.
        batch_size (int): Batching operations by triangles for improved speed
            and memory consumption. Default 10000.

    Returns:
        grid_3b (list): array-like list of shape
            (n_atoms, 3, n_basis_functions) where 3 refers to the three
            cartesian directions x, y, and z.
    """
    n_atoms = len(geom)
    L = len(knot_sequence) - 4
    coords, matrix, x_where, y_where = identify_ij_supercell(geom,
                                                             sup_geometry,
                                                             knot_sequence)
    lower_where, upper_where = sort_pairs(x_where, y_where)
    idx_ijk = generate_triplets(lower_where, upper_where)
    r_l, r_m, r_n, mask = get_triplet_distances(matrix,
                                                idx_ijk,
                                                knot_sequence)
    idx_ijk = idx_ijk[mask]
    # compute splines for all unique distances
    tuples_3b, idx_lmn = evaluate_triplet_derivs(
        r_l, r_m, r_n, basis_functions, knot_sequence)
    # batch by triangles to limit memory requirement (~1GB)
    force_grids = [[np.zeros((L, L, L)),
                    np.zeros((L, L, L)),
                    np.zeros((L, L, L))]
                   for _ in range(n_atoms)]  # array(n_atoms, 3, L, L, L)
    n_triangles = int(len(tuples_3b) / 4)
    n_batches = max(1, int(n_triangles / batch_size))
    indices = np.arange(n_batches) * batch_size * 4
    indices = np.append(indices, len(tuples_3b))  # tuple indices
    for batch_idx in range(n_batches):
        start = indices[batch_idx]  # tuple slice
        end = indices[batch_idx + 1]
        tri_0 = int(start / 4)  # four tuples per triangle
        tri_f = int(end / 4)
        drij_dr = distances.compute_direction_cosines(coords,
                                                      matrix,
                                                      idx_ijk[tri_0: tri_f, 0],
                                                      idx_ijk[tri_0: tri_f, 1],
                                                      n_atoms)
        drik_dr = distances.compute_direction_cosines(coords,
                                                      matrix,
                                                      idx_ijk[tri_0: tri_f, 0],
                                                      idx_ijk[tri_0: tri_f, 2],
                                                      n_atoms)
        drjk_dr = distances.compute_direction_cosines(coords,
                                                      matrix,
                                                      idx_ijk[tri_0: tri_f, 1],
                                                      idx_ijk[tri_0: tri_f, 2],
                                                      n_atoms)
        grids = spline_deriv_3b(tuples_3b[start: end],
                                idx_lmn[start: end],
                                drij_dr,
                                drik_dr,
                                drjk_dr,
                                L)
        for a in range(n_atoms):
            for c in range(3):
                force_grids[a][c] += grids[a][c]
    force_grids = [[symmetrize_3B(cart_grid)
                    for cart_grid in atom_grid]
                   for atom_grid in force_grids]  # enforce symmetry
    return force_grids


@jit(nopython=True, nogil=True)
def spline_deriv_3b(triangle_values,
                    idx_lmn,
                    drij_dr,
                    drik_dr,
                    drjk_dr,
                    L):
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
    M = L
    N = L
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


def identify_ij(geom, knot_sequence, supercell):
    """
    Args:
        geom (ase.Atoms)
        knot_sequence (np.ndarray)
        supercell (ase.Atoms)

    Returns:
        dist_matrix (np.ndarray): rectangular matrix of shape
            (n_atoms, n_supercell) where n_supercell is the number of
            atoms within the cutoff distance of any in-unit-cell atom.
        {i, j}_where (np.ndarray): unsorted list of atom indices
            over which to loop to obtain valid pair distances.
    """
    r_min = np.min(knot_sequence)
    r_max = np.max(knot_sequence)
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
    cut_matrix = dist_matrix[:n_geo, :]
    dist_mask = (cut_matrix > r_min) & (cut_matrix < r_max)
    i_where, j_where = np.where(dist_mask)
    return dist_matrix, i_where, j_where


def identify_ij_supercell(geom, sup_geometry, knot_sequence):
    """Copy of identify_ij() that produces a square distance matrix instead."""
    r_min = np.min(knot_sequence)
    r_max = np.max(knot_sequence)
    sup_positions = sup_geometry.get_positions()
    geo_positions = geom.get_positions()
    # mask atoms that aren't close to any unit-cell atom
    matrix = distance.cdist(geo_positions, sup_positions)
    weight_mask = matrix < r_max
    valids_mask = np.any(weight_mask, axis=0)
    coords = sup_positions[valids_mask, :]
    # reduced distance matrix
    matrix = distance.cdist(coords, coords)
    cut_mask = (matrix > r_min) & (matrix < r_max)
    x_where, y_where = np.where(cut_mask)
    return coords, matrix, x_where, y_where


def sort_pairs(x_where, y_where):
    """Sort atom indices from identify_ij() such that i < j."""
    idx_stacked = np.vstack((x_where, y_where)).T
    lower_where = np.min(idx_stacked, axis=1)
    upper_where = np.max(idx_stacked, axis=1)
    idx_sort = np.argsort(lower_where)
    lower_where = lower_where[idx_sort]
    upper_where = upper_where[idx_sort]
    return lower_where, upper_where


def generate_triplets(lower_where, upper_where):
    """
    Identify unique "i-j-j'" tuples by combining provided i-j pairs.

    Args:
        lower_where (np.ndarray): sorted "i" indices
        upper_where (np.ndarray): sorted "j" indices

    Returns:
        tuples_idx (np.ndarray): array of shape (n_triangles, 3)
    """
    # find unique values of i (sorted such that i < j)
    i_values, group_sizes = np.unique(lower_where, return_counts=True)
    # group j by values of i
    i_groups = np.array_split(upper_where, np.cumsum(group_sizes)[:-1])
    # generate j-k combinations
    jk_combinations = [np.array(np.meshgrid(i_groups[i],
                                            i_groups[i])).T.reshape(-1, 2)
                         for i in range(len(i_groups))]
    ijk_combinations = []
    for i in range(len(i_groups)):
        i_vector = np.ones(len(jk_combinations[i])) * i_values[i]
        combinations = np.insert(jk_combinations[i],
                                 0,
                                 i_vector,
                                 axis=1)
        ijk_combinations.append(combinations)
    tuples_idx = np.concatenate(ijk_combinations)
    return tuples_idx


def get_triplet_distances(distance_matrix, tuples, knot_sequence):
    """
    Identify i-j, i-k, and j-k pair distances from i-j-k tuples,
        distance matrix, and knot sequence for cutoffs.

    Args:
        distance_matrix (np.ndarray)
        tuples (np.ndarray): array of shape (n_triangles, 3)
        knot_sequence (np.ndarray)

    Returns:
        r_l (np.ndarray): vector of i-j distances.
        r_m (np.ndarray): vector of i-k distances.
        r_n (np.ndarray): vector of j-k distances.
        tuple_mask (np.ndarray): array of shape (n_triangles, 3)
            where triangles outside of cutoffs are ignored.
    """
    tuple_mask = (tuples[:, 1] < tuples[:, 2])
    tuples = tuples[tuple_mask]
    # extract distance tuples
    r_l = distance_matrix[tuples[:, 0], tuples[:, 1]]
    r_m = distance_matrix[tuples[:, 0], tuples[:, 2]]
    r_n = distance_matrix[tuples[:, 1], tuples[:, 2]]
    # mask by longest distance
    mask = (r_n > knot_sequence[0]) & (r_n < knot_sequence[-1])
    r_l = r_l[mask]
    r_m = r_m[mask]
    r_n = r_n[mask]
    return r_l, r_m, r_n, tuple_mask


def evaluate_triplet_distances(r_l, r_m, r_n, basis_functions, knot_sequence):
    """
    Identify non-zero basis functions for each point and call functions.

    Args:
        r_l (np.ndarray): vector of i-j distances.
        r_m (np.ndarray): vector of i-k distances.
        r_n (np.ndarray): vector of j-k distances.
        basis_functions (list): list of callable basis functions.
        knot_sequence (np.ndarray)

    Returns:
        tuples_3b (np.ndarray):
        idx_lmn (np.ndarray):
    """
    l_knots = knot_sequence
    m_knots = knot_sequence
    n_knots = knot_sequence
    # identify non-zero splines (tiling each distance x 4)
    r_l, idx_rl = bspline.find_spline_indices(r_l, l_knots)
    r_m, idx_rm = bspline.find_spline_indices(r_m, m_knots)
    r_n, idx_rn = bspline.find_spline_indices(r_n, n_knots)
    # evaluate splines per dimension
    L = len(l_knots) - 4
    M = len(m_knots) - 4
    N = len(n_knots) - 4
    n_tuples = len(r_l)
    tuples_3b = np.zeros((n_tuples, 3))  # array of basis-function values
    for l_idx in range(L):
        mask = (idx_rl == l_idx)
        points = r_l[mask]
        tuples_3b[mask, 0] = basis_functions[l_idx](points)
    for m_idx in range(M):
        mask = (idx_rm == m_idx)
        points = r_m[mask]
        tuples_3b[mask, 1] = basis_functions[m_idx](points)
    for n_idx in range(N):
        mask = (idx_rn == n_idx)
        points = r_n[mask]
        tuples_3b[mask, 2] = basis_functions[n_idx](points)
    idx_lmn = np.vstack([idx_rl, idx_rm, idx_rn]).T
    return tuples_3b, idx_lmn


def evaluate_triplet_derivs(r_l, r_m, r_n, basis_functions, knot_sequence):
    """
    Identify non-zero basis functions for each point and call functions.

    Args:
        r_l (np.ndarray): vector of i-j distances.
        r_m (np.ndarray): vector of i-k distances.
        r_n (np.ndarray): vector of j-k distances.
        basis_functions (list): list of callable basis functions.
        knot_sequence (np.ndarray)

    Returns:
        tuples_3b (np.ndarray):
        idx_lmn (np.ndarray):
    """
    l_knots = knot_sequence
    m_knots = knot_sequence
    n_knots = knot_sequence
    # identify non-zero splines (tiling each distance x 4)
    r_l, idx_rl = bspline.find_spline_indices(r_l, l_knots)
    r_m, idx_rm = bspline.find_spline_indices(r_m, m_knots)
    r_n, idx_rn = bspline.find_spline_indices(r_n, n_knots)
    # evaluate splines per dimension
    L = len(l_knots) - 4
    M = len(m_knots) - 4
    N = len(n_knots) - 4
    n_tuples = len(r_l)
    tuples_3b = np.zeros((n_tuples, 6))  # array of basis-function values
    for l_idx in range(L):
        mask = (idx_rl == l_idx)
        points = r_l[mask]
        tuples_3b[mask, 0] = basis_functions[l_idx](points)
        tuples_3b[mask, 3] = basis_functions[l_idx](points, nu=1)
    for m_idx in range(M):
        mask = (idx_rm == m_idx)
        points = r_m[mask]
        tuples_3b[mask, 1] = basis_functions[m_idx](points)
        tuples_3b[mask, 4] = basis_functions[m_idx](points, nu=1)
    for n_idx in range(N):
        mask = (idx_rn == n_idx)
        points = r_n[mask]
        tuples_3b[mask, 2] = basis_functions[n_idx](points)
        tuples_3b[mask, 5] = basis_functions[n_idx](points, nu=1)
    idx_lmn = np.vstack([idx_rl, idx_rm, idx_rn]).T
    return tuples_3b, idx_lmn


def symmetrize_3B(grid_3b):
    """
        Symmetrize 3D array with three mirror planes, enforcing permutational
            invariance with respect to i, j , and k indices.
            This allows us to avoid sorting, which is slow.
    """
    images = [grid_3b,
              grid_3b.transpose(0, 2, 1),
              grid_3b.transpose(1, 0, 2),
              grid_3b.transpose(1, 2, 0),
              grid_3b.transpose(2, 0, 1),
              grid_3b.transpose(2, 1, 0)]
    grid_3b = np.sum(images, axis=0)
    return grid_3b
