import numpy as np
from uf3.representation import knots
from scipy.spatial import distance
from scipy import interpolate


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
    idx = np.searchsorted(np.unique(knot_sequence), points, side='left') - 1
    idx = np.concatenate([idx, idx + 1, idx + 2, idx + 3])
    idx = idx[np.argsort(np.tile(np.arange(len(points)), 4))]
    points = np.repeat(points, 4)
    return points, idx


def evaluate_spline(positions, idx, knot_sequence):
    interval = knot_sequence[idx: idx + 5]
    b_knots = knots.knot_sequence_from_points(interval)
    bs_l = interpolate.BSpline(b_knots,
                               [0, 0, 0, 1, 0, 0, 0],
                               3,
                               extrapolate=False)
    return bs_l(positions)


def evaluate_3b(geometry, ext_geometry, knot_sequence):
    r_min = np.min(knot_sequence)
    r_max = np.max(knot_sequence)
    ext_positions = ext_geometry.get_positions()
    geo_positions = geometry.get_positions()
    # mask atoms that aren't close to any unit-cell atom
    distances = distance.cdist(geo_positions, ext_positions)
    weight_mask = distances < r_max
    valids_mask = np.any(weight_mask, axis=0)
    ext_positions = ext_positions[valids_mask, :]
    # reduced distance matrix
    n_geo = len(geo_positions)
    ext_distances = distance.cdist(ext_positions, ext_positions)
    distances = ext_distances[:n_geo, :]
    cut_mask = (distances > r_min) & (distances < r_max)
    x_where, y_where = np.where(cut_mask)
    # find pairs, i < j
    idx_stacked = np.vstack([x_where, y_where]).T
    lower_where = np.min(idx_stacked, axis=1)
    upper_where = np.max(idx_stacked, axis=1)
    idx_sort = np.argsort(lower_where)
    lower_where = lower_where[idx_sort]
    upper_where = upper_where[idx_sort]
    # find triples, i < j < k
    unique, counts = np.unique(lower_where, return_counts=True)
    bonds = np.array_split(upper_where, np.cumsum(counts)[:-1])
    pair_combinations = [np.array(np.meshgrid(bonds[i],
                                              bonds[i])).T.reshape(-1, 2)
                         for i in range(len(bonds))]
    for i in range(len(bonds)):
        center_idx = np.ones(len(pair_combinations[i])) * unique[i]
        pair_combinations[i] = np.insert(pair_combinations[i],
                                         0,
                                         center_idx,
                                         axis=1)
    tuples = np.concatenate(pair_combinations)
    tuple_mask = np.logical_and.reduce([tuples[:, 0] < tuples[:, 1],
                                        tuples[:, 1] < tuples[:, 2]])
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
    # evaluate splines
    v_grid = spline_3b(r_l, r_m, r_n, knot_sequence)
    return v_grid


def spline_3b(r_l, r_m, r_n, l_knots):
    m_knots = l_knots
    n_knots = l_knots
    # tile each distance vector for spline overlap
    r_l, idx_rl = find_spline_indices(r_l, l_knots)
    r_m, idx_rm = find_spline_indices(r_m, m_knots)
    r_n, idx_rn = find_spline_indices(r_n, n_knots)
    # evaluate splines per dimension
    L = len(l_knots)
    M = len(m_knots)
    N = len(n_knots)
    n_tuples = len(r_l)
    v_tuples = np.zeros((n_tuples, 3))
    for l_idx in range(L - 4):
        mask = (idx_rl == l_idx)
        positions = r_l[mask]
        v_tuples[mask, 0] = evaluate_spline(positions, l_idx, l_knots)
    for m_idx in range(M - 4):
        mask = (idx_rm == m_idx)
        positions = r_m[mask]
        v_tuples[mask, 1] = evaluate_spline(positions, m_idx, m_knots)
    for n_idx in range(N - 4):
        mask = (idx_rn == n_idx)
        positions = r_n[mask]
        v_tuples[mask, 2] = evaluate_spline(positions, n_idx, n_knots)

    # multiply spline values and arrange in 3D grid
    v_grid = np.zeros((L - 4, M - 4, N - 4))
    n_chunks = len(v_tuples) / 4
    try:
        chunks = np.split(np.arange(n_tuples), n_chunks)
    except ZeroDivisionError:
        return v_grid
    for chunk_slice in chunks:
        v_l = v_tuples[:, 0][chunk_slice][:, None, None]
        v_m = v_tuples[:, 1][chunk_slice][None, :, None]
        v_n = v_tuples[:, 2][chunk_slice][None, None, :]
        d_l = idx_rl[chunk_slice][:, None, None]
        d_m = idx_rm[chunk_slice][None, :, None]
        d_n = idx_rn[chunk_slice][None, None, :]
        v_grid[d_l, d_m, d_n] += v_l * v_m * v_n
    # enforce symmetry
    images = [v_grid,
              v_grid.transpose(0, 2, 1),
              v_grid.transpose(1, 0, 2),
              v_grid.transpose(1, 2, 0),
              v_grid.transpose(2, 0, 1),
              v_grid.transpose(2, 1, 0)]
    v_grid = np.sum(images, axis=0)
    return v_grid
