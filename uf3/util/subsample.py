import numpy as np
from scipy.spatial import distance


def farthest_point_sampling(data, max_samples=None, min_diff=0):
    """
    Subsample by iteratively selecting points with largest minimum
        distance from all previously-selected samples.
        e.g. on a range from 0 to 99: [0, 99, 49, 74, 24, 12, 36, ...]
        Two stopping criteria

    Args:
        data (np.ndarray)
        max_samples (int)
        min_diff (float): minimum distance between samples.

    Returns:
        indices (list): vector of subsample indices.
    """
    if data.ndim < 2:
        data = data[:, np.newaxis]
    dist_matrix = distance.cdist(data, data)

    if max_samples is None and min_diff == 0:
        return np.arange(len(data))
    elif max_samples is None or max_samples >= len(data) or max_samples < 1:
        max_samples = len(data)

    subsamples = np.array([np.argmin(data)])  # begin with lowest value
    while len(subsamples) < max_samples:
        dist_matrix[subsamples, :] = 0
        scores = np.min(dist_matrix[:, subsamples], axis=1)
        idx = np.argmax(scores)
        if np.max(scores) < min_diff:
            break
        subsamples = np.append(subsamples, idx)
    return subsamples
