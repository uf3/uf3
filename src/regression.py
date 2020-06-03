import numpy as np


def ridge_regression(A, y, lambda_=1):
    """
    Solves the Ax=y with ridge regression
    x = np.linalg.inv(A.T*A + alpha * I)*A.T*y

    Args:
        A (np.ndarray): representations.
        y (np.ndarray): energies.
        lambda_ (float): regularization parameter.

    Returns:
        beta (np.ndarray): coefficients.
        y_beta (np.ndarray): predicted energies.
        A_mean (np.ndarray): average representation; used for centering.
    """
    l, w = A.shape
    A_mean = np.mean(A, axis=0)
    kernel = np.dot(A.T, A) + lambda_ * np.eye(w)
    inverted_kernel = np.linalg.inv(kernel)
    beta = np.dot(np.dot(inverted_kernel, A.T), y)
    y_beta = np.dot(beta, A.T)
    return beta, y_beta, A_mean


def kernel_ridge_regression(x_flat, y_train, lambda_=1):
    """
    Args:
        x_flat (np.ndarray): representations.
        y_train (np.ndarray): energies.
        lambda_ (float): regularization parameter.

    Returns:
        beta (np.ndarray): coefficients.
        y_beta (np.ndarray): predicted energies.
        A_mean (np.ndarray): average representation; used for centering.
    """
    l, w = x_flat.shape
    y_offset = np.mean(y_train)
    y_centered = y_train - y_offset
    x_offset = np.mean(x_flat, axis=0)
    x_centered = x_flat - x_offset
    kernel_centered = np.dot(x_centered, x_centered.T)
    intermediate_inv = np.linalg.inv(np.dot(kernel_centered.T, kernel_centered)
                                     + lambda_ * np.eye(l))
    inverted_kernel = np.dot(intermediate_inv, kernel_centered)
    alpha = np.dot(inverted_kernel, y_centered)
    dxd = np.dot(x_centered.T, x_centered) + lambda_ * np.eye(w)
    dxd = np.linalg.inv(dxd)
    dxn = x_centered.T
    nxn = np.dot(x_centered, x_centered.T) + lambda_ * np.eye(l)
    beta = np.dot(np.dot(np.dot(dxd, dxn), nxn), alpha)
    y_beta = np.dot(beta, x_centered.T) + y_offset
    return beta, y_beta, x_offset, y_offset


def construct_D(x):
    """
    Construct partition matrix for regression with per-atom representations and
    total energies (i.e. one scalar per configuration).

    Args:
        x: nested list of lists containing per-atom representations,
            grouped by configuration

    Returns:
        D (np.ndarray): Partition matrix of M x N where M is the number of
            configurations and N is the number of atoms. The matrix is 1 where
            the j-th atom belongs the i-th configuration and 0 elsewhere.

    """
    M = len(x)  # number of configurations
    x_flat = np.concatenate(x)
    N = len(x_flat)  # total number of atoms across all configurations
    D = np.zeros((M, N))
    counter = 0
    for i, xi in enumerate(x):
        atoms = len(xi)
        D[int(i), counter:counter+atoms] = 1
        counter += atoms
    return D


def atomwise_uncenter(y_offset, D):
    """

    Args:
        y_offset (np.ndarray): offset in energies from centering.
        D (np.ndarray): Partition matrix of M x N where M is the number of
            configurations and N is the number of atoms. The matrix is 1 where
            the j-th atom belongs the i-th configuration and 0 elsewhere.

    Returns:
        y (np.ndarray): vector of uncentered energies per configuration
            after summation over atoms.
    """

    M_atomwise = np.sum(D, axis=1)
    y_normalized = np.divide(y_offset, M_atomwise)
    uncenter = lambda row: np.multiply(row, y_normalized)
    y_atomwise = np.apply_along_axis(uncenter, 0, D)
    y = np.sum(y_atomwise, axis=0)
    return y


def collected_ridge_regression(x_list, y, lambda_=1):
    """
    Solves ridge regression with per-atom representations and
    total energies (i.e. one scalar per configuration).

    Args:
        x_list (list): a list of collections of entries where each collection
            of entries is associated with one value in y. That value of y is
            effectively the sum of individual contributions from entries
            in that group.
        y (list): target scalar quantity e.g. total energy.
        lambda_ (float): regularization parameter.

    Returns:
        beta (np.ndarray): coefficients.
        y_beta (np.ndarray): predicted energies.
        A_offset (np.ndarray): offset in representations for centering.
        y_offset (np.ndarray): offset in energies for centering.
    """
    l, w = x_list[0].shape
    A = np.concatenate(x_list)
    A_offset = np.mean(A, axis=0)
    A_centered = A - A_offset
    kernel = np.dot(A_centered.T, A_centered) + lambda_ * np.eye(w)
    inverted_kernel = np.linalg.inv(kernel)

    y_offset = np.mean(y)
    y_centered = np.subtract(y, y_offset)
    D = construct_D(x_list)
    intermediate = np.dot(inverted_kernel, np.dot(D, A_centered).T)
    beta = np.dot(intermediate, y_centered)
    y_beta = np.dot(np.dot(beta, A_centered.T), D.T) + y_offset
    return beta, y_beta, A_offset, y_offset


def collected_kernel_ridge_regression(x_list, y, lambda_=1):
    """
    Solves ridge regression with per-atom representations and
    total energies (i.e. one scalar per configuration).

    Args:
        x_list (list): a list of collections of entries where each collection
            of entries is associated with one value in y. That value of y is
            effectively the sum of individual contributions from entries
            in that group.
        y (list): target scalar quantity e.g. total energy.
        lambda_ (float): regularization parameter.

    Returns:
        beta (np.ndarray): coefficients.
        y_beta (np.ndarray): predicted energies.
        A_offset (np.ndarray): offset in representations for centering.
        y_offset (np.ndarray): offset in energies for centering.
    """
    n_samples = len(x_list)
    x_flat = np.array([i for j in x_list for i in j])
    l, w = x_flat.shape
    y_offset = np.mean(y)
    y_centered = y - y_offset
    x_offset = np.mean(x_flat, axis=0)
    x_centered = x_flat - x_offset
    kernel_centered = np.dot(x_centered, x_centered.T)

    D = construct_D(x_list)
    intermediate = np.dot(np.dot(D, kernel_centered), D.T)
    intermediate_inv = np.linalg.inv(np.dot(intermediate.T, intermediate)
                                     + lambda_ * np.eye(n_samples))
    inverted_kernel = np.dot(intermediate_inv, intermediate)
    alpha = np.dot(np.dot(D.T, inverted_kernel), y_centered)
    dxd = np.dot(x_centered.T, x_centered) + lambda_ * np.eye(w)
    dxd = np.linalg.inv(dxd)
    dxn = x_centered.T
    nxn = np.dot(x_centered, x_centered.T) + lambda_ * np.eye(l)
    beta = np.dot(np.dot(np.dot(dxd, dxn), nxn), alpha)

    y_beta = np.dot(np.dot(beta, x_centered.T), D.T) + y_offset
    return beta, y_beta, x_offset, y_offset

# def ridge_regression(A, y, lambda_=1):
#     """
#     Solves the Ax=y with ridge regression
#     x = np.linalg.inv(A.T*A + alpha * I)*A.T*y
#     """
#     l, w = A.shape
#     A_offset = np.mean(A, axis=0)
#     A_centered = A - A_offset
#     kernel = np.dot(A_centered.T, A_centered) + lambda_ * np.eye(w)
#     inverted_kernel = np.linalg.inv(kernel)
#     y_offset = np.mean(y)
#     y_centered = y - y_offset
#     beta = np.dot(np.dot(inverted_kernel, A_centered.T), y_centered)
#     y_beta = np.dot(beta, A_centered.T) + y_offset
#     return beta, y_beta, A_offset, y_offset


# def ridge_predict(beta, A, A_offset, y_offset):
#     """
#     Solves the Ax=y with ridge regression
#     x = np.linalg.inv(A.T*A + alpha * I)*A.T*y
#     """
#     A_centered = A - A_offset
#     y_beta = np.dot(beta, A_centered.T) + y_offset
#     return y_beta
