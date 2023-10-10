"""
This module provides functions for generating regularizer matrices
for regularized least squares.
"""
from typing import List
import numpy as np


DEFAULT_REGULARIZER_GRID = dict(ridge_1b=1e-8,
                                ridge_2b=0.0,
                                ridge_3b=1e-5,
                                curve_2b=1e-8,
                                curve_3b=1e-8)


def get_regularizer_matrix(n_features: int,
                           ridge: float = 0.0,
                           curvature: float = 1.0
                           ) -> np.ndarray:
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


def get_penalty_matrix_2D(L: int,
                          M: int,
                          ridge: float = 0.0,
                          curvature: float = 1.0
                          ) -> np.ndarray:
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
        curvature (float): Local curvature regularization strength.

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


def get_penalty_matrix_3D(L: int,
                          M: int,
                          N: int,
                          ridge: float = 0.0,
                          curvature: float = 1.0,
                          periodic=(False, False, True),
                          ) -> np.ndarray:
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
        curvature (float): Local curvature regularization strength.
        periodic (list): periodicity in regularization for three dimensions.
            Default (False, False, True) for periodicity in angular space.

    Returns:
        matrix_3d (numpy.ndarray): square penalty matrix for linear
            least-squares of shape (L*M*N , L*M*N).
    """
    matrix_3d = np.zeros((L * M * N, L, M, N))
    idx = 0
    i_pbc, j_pbc, k_pbc = periodic

    for i in range(L):
        for j in range(M):
            for k in range(N):
                center_value = 2
                # i dimension
                if i_pbc:
                    matrix_3d[idx, (i - 1) % L, j, k] = -1
                    matrix_3d[idx, (i + 1) % L, j, k] = -1
                else:
                    if i > 0:  # lower bound
                        matrix_3d[idx, i - 1, j, k] = -1
                    else:
                        center_value = 1
                    if i + 1 < L:  # upper bound
                        matrix_3d[idx, i + 1, j, k] = -1
                    else:
                        center_value = 1
                # j dimension
                if j_pbc:
                    matrix_3d[idx, i, (j - 1) % M, k] = -1
                    matrix_3d[idx, i, (j + 1) % M, k] = -1
                else:
                    if j > 0:  # lower bound
                        matrix_3d[idx, i, j - 1, k] = -1
                    else:
                        center_value = 1
                    if j + 1 < M:
                        matrix_3d[idx, i, j + 1, k] = -1
                    else:
                        center_value = 1
                # k dimension
                if k_pbc:
                    matrix_3d[idx, i, j, (k - 1) % N] = -1
                    matrix_3d[idx, i, j, (k + 1) % N] = -1
                else:
                    if k > 0:  # lower bound
                        matrix_3d[idx, i, j, k - 1] = -1
                    else:
                        center_value = 1
                    if k + 1 < N:  # upper bound
                        matrix_3d[idx, i, j, k + 1] = -1
                    else:
                        center_value = 1

                matrix_3d[idx, i, j, k] = center_value

                idx += 1
    matrix_3d = matrix_3d.reshape(L * M * N, L * M * N) * curvature
    if ridge > 0:
        matrix_3d = matrix_3d + np.eye(L * M * N) * ridge
    return matrix_3d
