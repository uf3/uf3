"""
This module provides functions for generating regularizer matrices
for regularized least squares.
"""
from typing import List
import numpy as np


DEFAULT_REGULARIZER_GRID = dict(ridge_1b=1e-16,
                                ridge_2b=0.0,
                                ridge_3b=1e-10,
                                curve_2b=1e-16,
                                curve_3b=1e-16)


def get_ridge_penalty_matrix(n_features: int,
                             ) -> np.ndarray:
    """
    Generates L2 (ridge) regularization matrix for linear regression.

    Args:
        n_features (int): number of features in linear regression problem.
        ridge (float): L2 regularization strength (multiplicative)
            for ridge regression.

    Returns:
        matrix (numpy.ndarray): square matrix of size (n_features x n_features)
            with ridge penalty.
    """
    return np.eye(n_features)


def get_curvature_penalty_matrix_1D(n_features: int,
                                    ) -> np.ndarray:
    """
    Generates curvature regularization matrix in one dimension (2-body) for
    linear regression.
    
    The curvature penalty here applies to adjacent coefficients in one
    dimension.

    Args:
        n_features (int): number of features in linear regression problem.
        curvature (float): curvature regularization strength (multiplicative).
            Rule-of-thumb may be similar to ridge regression,
            e.g. optimized through cross-validation between 1e-3 to 1e3

    Returns:
        matrix (numpy.ndarray): square matrix of size (n_features x n_features)
            with curvature penalty.
    """
    reg_diag = np.eye(n_features) * -2
    reg_minus = np.eye(n_features, k=-1)
    reg_plus = np.eye(n_features, k=1)
    matrix = reg_diag + reg_minus + reg_plus
    matrix[0, 0] /= 2
    matrix[n_features - 1, n_features - 1] /= 2
    return matrix


def combine_regularizer_matrices(matrices: List) -> np.ndarray:
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
        matrices (list): list of matrices, e.g. for separate optimization
            objectives like A-A, A-B, B-B interactions.
            Does not need to be square.
            Number of columns is equal to the number of features.
            Number of rows is equal to the number of regularization conditions.

    Returns:
        full_matrix (numpy.ndarray): penalty matrix whose dimensions are equal
            to the sum of constituent dimensions.
    """
    # number of features and regularization conditions for each matrix
    n_reg_conds_i = [matrix.shape[0] for matrix in matrices]
    n_features_i = [matrix.shape[1] for matrix in matrices]

    # number of regularization conditions and features in combined matrix
    n_reg_conds = np.sum(n_reg_conds_i)
    n_features = np.sum(n_features_i)

    # initialize combined matrix
    full_matrix = np.zeros((n_reg_conds, n_features))
    origins_row = np.insert(np.cumsum(n_reg_conds_i), 0, 0)
    origins_col = np.insert(np.cumsum(n_features_i), 0, 0)

    # fill in combined matrix
    for i, matrix in enumerate(matrices):
        start_row = origins_row[i]
        end_row = origins_row[i + 1]
        start_col = origins_col[i]
        end_col = origins_col[i + 1]
        full_matrix[start_row:end_row, start_col:end_col] = matrix
    return full_matrix


def get_curvature_penalty_matrix_2D(L: int,
                                    M: int,
                                    flatten: bool = True,
                                    ) -> np.ndarray:
    """
    Generates curvature regularization matrix in two dimensions for linear
    regression.
 
    Curvature penalty here applies to coefficients that are spatially
    related in two dimensions that may not be adjacent after flattening
    for linear least-squares.

    Args:
        L (int): length of coefficient matrix before flattening.
        M (int): width of coefficient matrix before flattening.
        curvature (float): Local curvature regularization strength.
        flatten (bool): whether to flatten each row to 1D.

    Returns:
        matrix_2d (numpy.ndarray): square curvature penalty matrix for linear
            least-squares of shape (L*M , L*M) or (L*M, L, M) if not flattened.
    """
    matrix_2d = np.zeros((L * M, L, M))
    idx = 0
    for i in range(L):
        for j in range(M):
            if i > 0:
                matrix_2d[idx, i - 1, j] = 1
            if i + 1 < L:
                matrix_2d[idx, i + 1, j] = 1
            if j > 0:
                matrix_2d[idx, i, j - 1] = 1
            if j + 1 < M:
                matrix_2d[idx, i, j + 1] = 1
            center_value = -np.sum(matrix_2d[idx])
            matrix_2d[idx, i, j] = center_value
            idx += 1
    if flatten:
        matrix_2d = matrix_2d.reshape(L * M, L * M)
    return matrix_2d


def get_curvature_penalty_matrix_3D(L: int,
                                    M: int,
                                    N: int,
                                    flatten: bool = True,
                                    ) -> np.ndarray:
    """
    Generates curvature regularization matrix in three dimensions (2-body) for
    linear regression.
 
    Curvature penalty here applies to coefficients that are spatially
    related in three dimensions that may not be adjacent after flattening
    for linear least-squares.

    Args:
        L (int): length of coefficient matrix before flattening.
        M (int): width of coefficient matrix before flattening.
        N (int): depth of coefficient matrix before flattening.
        curvature (float): Local curvature regularization strength.
        flatten (bool): whether to flatten each row to 1D.

    Returns:
        matrix_3d (numpy.ndarray): square curvature penalty matrix for linear
            least-squares of shape (L*M , L*M*N) or (L*M, L, M, N) if not
            flattened.
    """
    matrix_3d = np.zeros((L * M * N, L, M, N))
    idx = 0

    for i in range(L):
        for j in range(M):
            for k in range(N):
                # i dimension
                if i > 0:  # lower bound
                    matrix_3d[idx, i - 1, j, k] = 1
                if i + 1 < L:  # upper bound
                    matrix_3d[idx, i + 1, j, k] = 1
                # j dimension
                if j > 0:  # lower bound
                    matrix_3d[idx, i, j - 1, k] = 1
                if j + 1 < M:
                    matrix_3d[idx, i, j + 1, k] = 1
                # k dimension
                if k > 0:  # lower bound
                    matrix_3d[idx, i, j, k - 1] = 1
                if k + 1 < N:  # upper bound
                    matrix_3d[idx, i, j, k + 1] = 1
                center_value = -np.sum(matrix_3d[idx])
                matrix_3d[idx, i, j, k] = center_value
                idx += 1
    if flatten:
        matrix_3d = matrix_3d.reshape(L * M * N, L * M * N)
    return matrix_3d
