import numpy as np


class Regularizer:
    """
    -Manage regularization parameters
    -Generate L2 (ridge) penalty matrices for regression
    -Generate curvature penalty matrices for regression
    -Arrange matrices by chemical interaction
    -Manage feature indices for fixed coefficients
    """
    def __init__(self,
                 regularizer_sizes=None,
                 ridge=1e-6,
                 curvature=1e-5,
                 onebody=1e-4):
        """
        Args:
            regularizer_sizes (list): List of integers corresponding to
                the size of feature groups to regularize separately.
                e.g. regularizer_size for a unary system:
                    [10, 1], corresponding to A-A features alongside the
                    elemental one-body term.
                e.g. regularizer_size for a binary system:
                    [10, 10, 10, 2], corresponding to A-A, A-B, and B-B
                    features alongside the elemental one-body terms.
            curvature (float): curvature regularization strength.
                Rule-of-thumb may be similar to ridge regression,
                e.g. optimized through cross-validation between 1e-3 to 1e3
            ridge (float): L2 regularization strength (multiplicative)
                for ridge regression.
            onebody (bool, float): If False or None, process all
                entries in regularizer_sizes with ridge and curvature.
                Otherwise, treat the last entry as the one-body term,
                ignoring curvature penalty. If a float is provided,
                apply as a separate L2 regularization strength. Providing a
                larger value discourages larger values in onebody coefficients.
        """
        self.regularizer_sizes = np.array(regularizer_sizes, dtype=int)
        self.onebody = onebody
        self.ridge = ridge
        self.curvature = curvature

    @property
    def matrix(self):
        if self.regularizer_sizes is not None:
            n_segments = len(self.regularizer_sizes)
            ridge_strengths = np.ones(n_segments) * self.ridge
            curv_strengths = np.ones(n_segments) * self.curvature
            if self.onebody is not False and self.onebody is not None:
                curv_strengths[-1] = 0
            if isinstance(self.onebody, (int, float, np.floating)):
                ridge_strengths[-1] = self.onebody  # composition segment
            matrices = []
            for i in range(n_segments):
                n_features = self.regularizer_sizes[i]
                curv_strength = curv_strengths[i]
                r_matrix = get_regularizer_matrix(n_features,
                                                  ridge=ridge_strengths[i],
                                                  curvature=curv_strength)
                matrices.append(r_matrix)
            combined_matrix = combine_regularizer_matrices(matrices)
            return combined_matrix
        else:
            raise ValueError("regularizer_sizes not specified.")

    def get_fixed(self,
                  value=0,
                  onebody=False,
                  upper_bounds=True,
                  lower_bounds=False):
        feature_chunksizes = self.regularizer_sizes
        n_features = np.sum(feature_chunksizes)
        indices = []
        if onebody:
            # trailing coefficients (element-specific one-body terms)
            for i in range(feature_chunksizes[-1]):
                indices.append(n_features - 1 - i)
        feature_chunksizes = feature_chunksizes[:-1]
        bounds = np.insert(np.cumsum(feature_chunksizes) - 1, 0, 0)
        lower_idxs = bounds[:-1]
        upper_idxs = bounds[1:]
        if lower_bounds:  #
            for idx in lower_idxs:
                indices.append(idx)
        if upper_bounds:
            for idx in upper_idxs:
                indices.append(idx)
        values = np.ones(len(indices)) * value
        fixed = np.vstack([indices, values]).T
        return fixed.astype(int)


def get_regularizer_matrix(n_features, ridge=0, curvature=1):
    """
    Generates additive regularization matrix for linear regression
        using curvature penalty and/or L2 (ridge) penalty.
        The curvature penalty here applies to adjacent coefficients
        in one dimension.

    Args:
        n_features (int): number of features in linear regression problem.
        curvature (float): curvature regularization strength (multiplicative).
            Rule-of-thumb may be similar to ridge regression,
            e.g. optimized through cross-validation between 1e-3 to 1e3
        ridge (float): L2 regularization strength (multiplicative)
            for ridge regression.

    Returns:
        matrix (numpy.ndarray): square matrix of size (n_features x n_features)
            with ridge and curvature penalty a.k.a. fused ridge regression.
    """
    reg_diag = np.eye(n_features) * 2 * curvature
    reg_minus = np.eye(n_features, k=-1) * -curvature
    reg_plus = np.eye(n_features, k=1) * -curvature
    matrix = reg_diag + reg_minus + reg_plus
    matrix[0, 0] /= 2
    matrix[n_features - 1, n_features - 1] /= 2
    if ridge > 0:
        matrix = matrix + np.eye(n_features) * ridge
    return matrix


def combine_regularizer_matrices(matrices):
    """
    Combine square penalty matrices.
        Example:
            [X---      [Y-      [Z--        [X--------
             -X--       -Y]      -Z-         -X-------
             --X-                --Z]        --X------
             ---X]                           ---X-----
            (4x4)      (2x2)    (3x3)   ->   ----Y----  (9x9)
                                             -----Y---
                                             ------Z--
                                             -------Z-
                                             --------Z]
    Args:
        matrices (list): square matrices, e.g. for separate optimization
            objectives like A-A, A-B, B-B interactions.

    Returns:
        full_matrix (numpy.ndarray): square penalty matrix equal in length
            to the sum of constituent lengths.
    """
    sizes = [len(m) for m in matrices]
    n_features = int(np.sum(sizes))
    full_matrix = np.zeros((n_features, n_features))
    origins = np.insert(np.cumsum(sizes), 0, 0)
    for i, matrix in enumerate(matrices):
        start = origins[i]
        end = origins[i + 1]
        full_matrix[start:end, start:end] = matrix
    return full_matrix


def get_penalty_matrix_2D(L, M, ridge=0, curvature=1):
    """
    Generates additive regularization matrix for linear regression
        using curvature penalty and/or L2 (ridge) penalty.
        Curvature penalty here applies to coefficients that are spatially
        related in two dimensions that may not be adjacent after flattening
        for linear least-squares.

    Args:
        L (int): length of coefficient matrix before flattening.
        M (int): width of coefficient matrix before flattening.
        ridge (float): L2 (ridge) regularization strength.

    Returns:
        matrix_2d (numpy.ndarray): square penalty matrix for linear
            least-squares of shape (L*M , L*M).
    """
    matrix_2d = np.zeros((L * M, L, M))
    idx = 0
    for i in range(L):
        for j in range(M):
            if any([i == 0, i == L - 1, j == 0, j == M - 1]):
                matrix_2d[idx, i, j] = 1
            else:
                matrix_2d[idx, i, j] = 2

            if i > 0:
                matrix_2d[idx, i - 1, j] = -1
            if i + 1 < L:
                matrix_2d[idx, i + 1, j] = -1
            if j > 0:
                matrix_2d[idx, i, j - 1] = -1
            if j + 1 < M:
                matrix_2d[idx, i, j + 1] = -1
            idx += 1
    matrix_2d = matrix_2d.reshape(L * M, L * M) * curvature
    if ridge > 0:
        matrix_2d = matrix_2d + np.eye(L * M) * ridge
    return matrix_2d


def get_penalty_matrix_3D(L, M, N, ridge=0, curvature=1):
    """
    Generates additive regularization matrix for linear regression
        using curvature penalty and/or L2 (ridge) penalty.
        Curvature penalty here applies to coefficients that are spatially
        related in three dimensions that may not be adjacent after flattening
        for linear least-squares

    Args:
        L (int): length of coefficient matrix before flattening.
        M (int): width of coefficient matrix before flattening.
        N (int): depth of coefficient matrix before flattening.
        ridge (float): L2 (ridge) regularization strength.

    Returns:
        matrix_3d (numpy.ndarray): square penalty matrix for linear
            least-squares of shape (L*M*N , L*M*N).
    """
    matrix_3d = np.zeros((L * M * N, L, M, N))
    idx = 0
    for i in range(L):
        for j in range(M):
            for k in range(N):
                if any([i == 0, i == L - 1, j == 0, j == M - 1, k == 0,
                        k == N - 1]):
                    matrix_3d[idx, i, j, k] = 1
                else:
                    matrix_3d[idx, i, j, k] = 2

                if i > 0:
                    matrix_3d[idx, i - 1, j, k] = -1
                if i + 1 < L:
                    matrix_3d[idx, i + 1, j, k] = -1
                if j > 0:
                    matrix_3d[idx, i, j - 1, k] = -1
                if j + 1 < M:
                    matrix_3d[idx, i, j + 1, k] = -1
                if k > 0:
                    matrix_3d[idx, i, j, k - 1] = -1
                if k + 1 < N:
                    matrix_3d[idx, i, j, k + 1] = -1
                idx += 1
    matrix_3d = matrix_3d.reshape(L * M * N, L * M * N) * curvature
    if ridge > 0:
        matrix_3d = matrix_3d + np.eye(L * M * N) * ridge
    return matrix_3d