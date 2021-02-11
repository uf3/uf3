import warnings
import itertools
import numpy as np
import ase
from numba import jit
from uf3.representation import knots
from uf3.representation import bspline
from uf3.representation import distances
from scipy.spatial import distance
from scipy import interpolate


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
    triangle_values, idx_rl, idx_rm, idx_rn = evaluate_triplet_distances(
        r_l, r_m, r_n, basis_functions, knot_sequence)
    energy = evaluate_3b(triangle_values, idx_rl, idx_rm, idx_rn, grid)
    return energy


def get_forces_3B(grid,
                  geom,
                  supercell,
                  knot_sequence,
                  basis_functions):
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
    triangle_values, idx_rl, idx_rm, idx_rn = evaluate_triplet_derivs(
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
                               idx_rl,
                               idx_rm,
                               idx_rn,
                               drij_dr,
                               drik_dr,
                               drjk_dr,
                               grid)
    return forces


def featurize_energy_3B(geom, supercell, knot_sequence, basis_functions):
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
    tuples_3b, idx_rl, idx_rm, idx_rn = evaluate_triplet_distances(
        r_l, r_m, r_n, basis_functions, knot_sequence)
    # arrange spline values into grid - BOTTLENECK
    L = len(knot_sequence) - 4
    grid_3b = arrange_3B(tuples_3b, idx_rl, idx_rm, idx_rn, L)
    # enforce symmetry
    grid_3b = symmetrize_3B(grid_3b)
    return grid_3b


def identify_ij(geom, knot_sequence, supercell):
    r_min = np.min(knot_sequence)
    r_max = np.max(knot_sequence)
    sup_positions = supercell.get_positions()
    geo_positions = geom.get_positions()
    # mask atoms that aren't close to any unit-cell atom
    dist_matrix = distance.cdist(geo_positions, sup_positions)
    cutoff_mask = dist_matrix < r_max
    cutoff_mask = np.any(cutoff_mask, axis=0)
    sup_positions = sup_positions[cutoff_mask, :]  # reduced distance matrix
    # enforce distance cutoffs
    n_geo = len(geo_positions)
    dist_matrix = distance.cdist(sup_positions, sup_positions)
    dist_matrix = dist_matrix[:n_geo, :]
    dist_mask = (dist_matrix > r_min) & (dist_matrix < r_max)
    i_where, j_where = np.where(dist_mask)
    return dist_matrix, i_where, j_where


@jit(nopython=True, nogil=True)
def arrange_3B(triangle_values, idx_rl, idx_rm, idx_rn, L):
    # multiply spline values and arrange in 3D grid
    M = L
    N = L
    grid = np.zeros((L, M, N))
    n_values = len(triangle_values)
    n_triangles = int(n_values / 4)
    for triangle_idx in range(n_triangles):
        idx_l = idx_rl[triangle_idx * 4]
        idx_m = idx_rm[triangle_idx * 4]
        idx_n = idx_rn[triangle_idx * 4]
        values = triangle_values[triangle_idx * 4: (triangle_idx + 1) * 4, :]
        # each triangle influences 4 x 4 x 4 = 64 basis functions

        # Broadcasting
        # value = (values[:, 0][:, None, None]
        #          * values[:, 1][None, :, None]
        #          * values[:, 2][None, None, :])
        # grid[idx_l:idx_l+4, idx_m:idx_m+4, idx_n:idx_n+4] += value

        # Einsum
        # value = np.einsum('i,j,k->ijk',
        #                   values[:, 0],
        #                   values[:, 1],
        #                   values[:, 2],
        #                   out=grid[idx_l:idx_l+4,
        #                            idx_m:idx_m+4,
        #                            idx_n:idx_n+4])

        # for loops
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    value = values[i, 0] * values[j, 1] * values[k, 2]
                    grid[idx_l + i, idx_m + j, idx_n + k] += value
    return grid



@jit(nopython=True, nogil=True)
def evaluate_3b(triangle_values, idx_rl, idx_rm, idx_rn, grid):
    # multiply spline values and arrange in 3D grid
    energy = 0.0
    n_values = len(triangle_values)
    n_triangles = int(n_values / 4)
    for triangle_idx in range(n_triangles):
        idx_l = idx_rl[triangle_idx * 4]
        idx_m = idx_rm[triangle_idx * 4]
        idx_n = idx_rn[triangle_idx * 4]
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
                      idx_rl,
                      idx_rm,
                      idx_rn,
                      drij_dr,
                      drik_dr,
                      drjk_dr,
                      grid):
    # multiply spline values and arrange in 3D grid
    n_atoms = len(drij_dr)
    forces = np.zeros((n_atoms, 3))

    n_values = len(triangle_values)
    n_triangles = int(n_values / 4)
    for triangle_idx in range(n_triangles):
        idx_l = idx_rl[triangle_idx * 4]
        idx_m = idx_rm[triangle_idx * 4]
        idx_n = idx_rn[triangle_idx * 4]
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


# def partition_triangles(triangle_values, idx_l, idx_m, idx_n, L):
#     M = L
#     N = L
#     n_values = len(triangle_values)
#     n_triangles = int(n_values / 4)
#     # each triangle has 12 (4 x 3) associated basis functions.
#     # convert 3D coordinates into flat 1D indices (L * M * N)
#     idx_flat = idx_l[::4] * (M - 3) * (N - 3) + idx_m[::4] * (N - 3) + idx_n[::4]
#     offsets = np.tile([0, 1, 2, 3], len(idx_flat))
#     idx_sort = np.repeat(np.argsort(idx_flat) * 4, 4) + offsets
#     # identify number of triangles per flat index
#     n_groups = (L -3) * (M -3) * (N -3)
#     group_sizes, _ = np.histogram(idx_flat, bins=np.arange(n_groups + 1))
#     group_sizes *= 4
#     # partition triangles by flat index
#     triangle_groups = np.array_split(triangle_values[idx_sort],
#                                      np.cumsum(group_sizes)[:-1])
#     return triangle_groups
#
#
# @jit(nopython=True, cache=True, nogil=True)
# def evaluate_local_triangles(triangle_group):
#     # triangle information is grouped into 4 tuples of 3
#     # corresponding to four basis functions for each of 3 dimensions
#     # goal is to find the cartesian product, yielding 64 values.
#     grid_data = np.zeros(64)
#     n_values = len(triangle_group)
#     n_triangles = int(n_values / 4)
#     for triangle_idx in range(n_triangles):
#         values = triangle_group[triangle_idx * 4: (triangle_idx + 1) * 4, :]
#         # each triangle influences 4 x 4 x 4 = 64 basis functions
#         for i in range(4):
#             for j in range(4):
#                 for k in range(4):
#                     idx_flat = i * 16 + j * 4 + k
#                     value = values[i, 0] * values[j, 1] * values[k, 2]
#                     grid_data[idx_flat] += value
#     return grid_data
#
#
# @jit(nopython=True, cache=True, nogil=True)
# def unravel_3d(index, shape):
#     sizes = np.zeros(3, dtype=np.int64)
#     result = np.zeros(3, dtype=np.int64)
#     sizes[-1] = 1
#     for i in range(1, -1, -1):
#         sizes[i] = sizes[i + 1] * shape[i + 1]
#     remainder = index
#     for i in range(3):
#         result[i] = remainder // sizes[i]
#         remainder %= sizes[i]
#     return result
#
#
# @jit(nopython=True, cache=True, nogil=True)
# def arrange_subgrids(subgrids, L):
#     M = L
#     N = L
#     grid_overall = np.zeros((L, M, N))
#     n_subgrids = len(subgrids)
#     for idx_subgrid in range(n_subgrids):
#         l, m, n = unravel_3d(idx_subgrid, (L - 3, M - 3, N - 3))
#         for i in range(4):
#             for j in range(4):
#                 for k in range(4):
#                     idx_flat = i * 16 + j * 4 + k
#                     value = subgrids[idx_subgrid][idx_flat]
#                     grid_overall[l + i, m + j, n + k] += value
#     return grid_overall


# @jit(nopython=True, cache=True, nogil=True)
# def spline_3b(triangle_values, idx_rl, idx_rm, idx_rn, knot_sequence):
#     # multiply spline values and arrange in 3D grid
#     L = len(knot_sequence) - 4
#     M = len(knot_sequence) - 4
#     N = len(knot_sequence) - 4
#
#     grid = np.zeros((L, M, N))
#     n_values = len(triangle_values)
#     n_triangles = int(n_values / 4)
#     for triangle_idx in range(n_triangles):
#         idx_l = idx_rl[triangle_idx * 4]
#         idx_m = idx_rm[triangle_idx * 4]
#         idx_n = idx_rn[triangle_idx * 4]
#         values = triangle_values[triangle_idx * 4: (triangle_idx + 1) * 4, :]
#         # each triangle influences 4 x 4 x 4 = 64 basis functions
#         for i in range(4):
#             for j in range(4):
#                 for k in range(4):
#                     # idx_flat = i * 16 + j * 4 + k
#                     value = values[i, 0] * values[j, 1] * values[k, 2]
#                     grid[idx_l + i, idx_m + j, idx_n + k] += value
#     return grid


@jit(nopython=True, nogil=True)
def spline_deriv_3b(triangle_values,
                    idx_rl,
                    idx_rm,
                    idx_rn,
                    drij_dr,
                    drik_dr,
                    drjk_dr,
                    L):
    # multiply spline values and arrange in 3D grid
    M = L
    N = L
    n_atoms = len(drij_dr)
    force_grids = [(np.zeros((L, M, N)),
                    np.zeros((L, M, N)),
                    np.zeros((L, M, N)))
                   for _ in range(n_atoms)]

    n_values = len(triangle_values)
    n_triangles = int(n_values / 4)
    for a in range(n_atoms):  # atom index
        for c in range(3):  # cartesian directions
            for triangle_idx in _:
                idx_l = idx_rl[triangle_idx * 4]
                idx_m = idx_rm[triangle_idx * 4]
                idx_n = idx_rn[triangle_idx * 4]
                v = triangle_values[triangle_idx * 4: (triangle_idx + 1) * 4, :]
                # each triangle influences 4 x 4 x 4 = 64 basis functions

                dij = drij_dr[a, c, triangle_idx]
                dik = drik_dr[a, c, triangle_idx]
                djk = drjk_dr[a, c, triangle_idx]
                for i in range(4):
                    for j in range(4):
                        for k in range(4):
                            val = (v[i, 3] * v[j, 1] * v[k, 2] * dij +
                                   v[i, 0] * v[j, 4] * v[k, 2] * dik +
                                   v[i, 0] * v[j, 1] * v[k, 5] * djk)
                            force_grids[a][c][idx_l+i, idx_m+j, idx_n+k] += val
    return force_grids


def sort_pairs(x_where, y_where):
    # find pairs, i < j
    idx_stacked = np.vstack((x_where, y_where)).T
    lower_where = np.min(idx_stacked, axis=1)
    upper_where = np.max(idx_stacked, axis=1)
    idx_sort = np.argsort(lower_where)
    lower_where = lower_where[idx_sort]
    upper_where = upper_where[idx_sort]
    return lower_where, upper_where


def generate_triplets(lower_where, upper_where):
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


def get_triplet_distances(ext_distances, tuples, knot_sequence):
    # tuple_mask = np.logical_and.reduce([tuples[:, 0] < tuples[:, 1],
    #                                     tuples[:, 1] < tuples[:, 2]])
    tuple_mask = (tuples[:, 1] < tuples[:, 2])
    tuples = tuples[tuple_mask]
    # extract distance tuples
    r_l = ext_distances[tuples[:, 0], tuples[:, 1]]
    r_m = ext_distances[tuples[:, 0], tuples[:, 2]]
    r_n = ext_distances[tuples[:, 1], tuples[:, 2]]
    # mask by longest distance (hotfix)
    mask = (r_n > knot_sequence[0]) & (r_n < knot_sequence[-1])
    r_l = r_l[mask]
    r_m = r_m[mask]
    r_n = r_n[mask]
    return r_l, r_m, r_n, tuple_mask


def evaluate_triplet_distances(r_l, r_m, r_n, basis_functions, knot_sequence):
    """
    Identify non-zero basis functions for each point and call functions.

    Args:
        r_l (np.ndarray):
        r_m (np.ndarray):
        r_n (np.ndarray):
        basis_functions:
        knot_sequence:

    Returns:

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
    return tuples_3b, idx_rl, idx_rm, idx_rn


def evaluate_triplet_derivs(r_l, r_m, r_n, basis_functions, knot_sequence):
    """
    Identify non-zero basis functions for each point and call functions.

    Args:
        r_l (np.ndarray):
        r_m (np.ndarray):
        r_n (np.ndarray):
        basis_functions:
        knot_sequence:

    Returns:

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
    return tuples_3b, idx_rl, idx_rm, idx_rn


def symmetrize_3B(grid_3b):
    """
        Symmetrize 3D array with three mirror planes, enforcing permutational
            invariance with respect to i, j , and k indices.This allows us to avoid
             sorting, which is slow.
        """
    images = [grid_3b,
              grid_3b.transpose(0, 2, 1),
              grid_3b.transpose(1, 0, 2),
              grid_3b.transpose(1, 2, 0),
              grid_3b.transpose(2, 0, 1),
              grid_3b.transpose(2, 1, 0)]
    grid_3b = np.sum(images, axis=0)
    return grid_3b


def featurize_force_3B(geom, sup_geometry, knot_sequence, basis_functions):
    n_atoms = len(geom)
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
    # TODO: extract method
    lower_where, upper_where = sort_pairs(x_where, y_where)
    tuples = generate_triplets(lower_where, upper_where)
    r_l, r_m, r_n, mask = get_triplet_distances(matrix,
                                                tuples,
                                                knot_sequence)
    # evaluate splines
    tuples_3b, idx_rl, idx_rm, idx_rn = evaluate_triplet_derivs(
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
    L = len(knot_sequence) - 4
    grids = spline_deriv_3b(tuples_3b,
                            idx_rl,
                            idx_rm,
                            idx_rn,
                            drij_dr,
                            drik_dr,
                            drjk_dr,
                            L)
    grid_3b = [[symmetrize_3B(cart_grid)
                for cart_grid in atom_grid]
               for atom_grid in grids]  # enforce symmetry
    return grid_3b
