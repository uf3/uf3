import numpy as np
from ufpotential.regression import regularize


class WeightedLinearModel:
    def __init__(self,
                 weights=None,
                 regularizer_sizes=None,
                 ridge=1e-6,
                 curvature=1e-5,
                 onebody=1e-4):
        if isinstance(weights, (list, np.ndarray)):
            self.weights = weights
        self.coefficients = None
        self.regularizer = regularize.Regularizer(
            regularizer_sizes=regularizer_sizes,
            ridge=ridge,
            curvature=curvature,
            onebody=onebody)

    def fit(self, x, y):
        solution, predictions = weighted_least_squares(x,
                                                       y,
                                                       self.weights,
                                                       self.regularizer)
        self.coefficients = solution

    def predict(self, x):
        predictions = np.dot(x, self.coefficients)
        return predictions

    def score(self, x, y):
        predictions = self.predict(x)
        rms = np.sqrt(np.mean(np.subtract(y, predictions)**2))
        return rms


def linear_least_squares(A, y, regularizer_matrix=None):
    """
    Solves the linear least-squares problem Ax=y with L2 (ridge)
        and/or curvature penalty.

    Args:
        A (np.ndarray): input matrix of shape (n_samples, n_features).
        y (np.ndarray): output vector of length n_samples.
        regularizer_matrix (np.ndarray): e.g. matrix generated using
            ufpotentials.regression.regularize.get_curvature_penalty_matrix


    Returns:
        solution (np.ndarray): coefficients.
        predictions (np.ndarray): predictions.
    """
    if regularizer_matrix is None:
        regularizer_matrix = 0
    kernel = np.dot(A.T, A) + regularizer_matrix
    inverted_kernel = np.linalg.inv(kernel)
    solution = np.dot(np.dot(inverted_kernel, A.T), y)
    predictions = np.dot(A, solution)
    return solution, predictions


def weighted_least_squares(A, y, weights=None, regularizers=None):
    """
    Solves the linear least-squares problem [A,B,...]x = [a,b,...]
        with optional square regularizer matrix and relative weighting.
        Regardless of weighting, error contributions are additionally
        normalized by the respective sample standard deviations,
        becoming unitless errors.

    Args:
        A (list of np.ndarray): input matrices.
            e.g. [energy inputs (100 x 20), force inputs (3200 x 20)]
        y (list of np.ndarray): output vectors.
            e.g. [energy outputs (100), force outputs (3200)]
        weights (list): relative weights (optional).
            e.g. [0.3, 0.7], weighing forces more than energies.
        regularizers (np.ndarray, list): matrix or list of matrices, e.g. from
            ufpotentials.regression.regularize.get_curvature_penalty_matrix

    Returns:
        solution (np.ndarray): coefficients.
        predictions (list of np.ndarray): predictions.
    """
    if regularizers is None:
        regularizers = ()
    elif isinstance(regularizers, np.ndarray):
        n_feats = len(A[0][0])
        if regularizers.shape == (n_feats, n_feats):
            regularizers = [regularizers]
        else:
            raise ValueError(
                "Expected regularizer shape: {} x {}".format(n_feats, n_feats))
    if weights is None:
        weights = np.ones(len(A)) / len(A)
    else:
        if len(weights) != len(A):
            raise ValueError(
                'Number of weights does not match number of arrays.')
        if not np.all(np.positive(weights)) or np.sum(weights) <= 0:
            raise ValueError('Negative or zero weights provided.')
        weights = weights / np.sum(weights)  # normalize
    xTx_list = [weight / np.std(target) / len(target) * np.dot(x.T, x)
           for weight, x, target in zip(weights, A, y)]
    xTx = np.sum(xTx_list, axis=0) + np.sum(regularizers, axis=0)
    inverted_xTx = np.linalg.inv(xTx)
    xTy_list = [weight / np.std(target) / len(target) * np.dot(x.T, target)
               for weight, x, target in zip(weights, A, y)]
    xTy = np.sum(xTy_list, axis=0) + 0
    solution = np.dot(inverted_xTx, xTy)
    predictions = [np.dot(x, solution) for x in A]
    return solution, predictions


def weight_regression_gridsearch(x, fx, y, fy, kappa, n_folds=5,
                                 logmin=-3, logmax=3, resolution=11):
    """
    Convenience function for grid search and fit with energies and forces.
        Samples "resolution" times from 10**logmin to 10**logmax
        for both ridge penalty and curvature penalty magnitudes.

    Args:
        x (np.ndarray): energy descriptor.
        fx (np.ndarray): force descriptor.
        y (np.ndarray): energy reference values.
        fy (np.ndarray): force reference values.
        kappa (float): weighting parameter between 0 and 1 e.g. 1 ignores forces.
        n_folds (int): number of folds for cross validation.
        logmin (float): power of 10 of lower bound for grid search.
        logmax (float): power of 10 of upper bound for grid search.
        resolution: grid search samples per axis.

    Returns:
        coefficients (np.ndarray): fit coefficients from linear least-squares.
        score (float): weighted RMSE.
    """
    n_features = x.shape[1]
    if x.shape[1] != fx.shape[1]:
        raise ValueError('Mismatch in basis between energy and force inputs.')
    res = {}
    for lambda1 in np.logspace(logmin, logmax, resolution):
        for lambda2 in np.logspace(logmin, logmax, resolution):
            train_rmse, cv_rmse = weighted_cross_validation(x, fx, y, fy,
                                                       kappa,
                                                       n_folds=n_folds,
                                                       lambda1=lambda1,
                                                       lambda2=lambda2)
            res[(lambda1, lambda2)] = cv_rmse
    lambda_tuples = list(res.keys())
    lambda1, lambda2 = sorted(lambda_tuples, key=lambda k: res[k])[0]
    reg = regularize.get_regularizer_matrix(n_features,
                                            ridge=lambda1,
                                            curvature=lambda2)
    coefficients = weighted_least_squares([x, fx], [y, fy], [kappa, 1 - kappa],
                                          regularizers=reg)
    pt = np.dot(x, coefficients)
    gt = np.dot(fx, coefficients)
    score = two_class_weighted_score(y, fy, pt, gt, kappa)
    return coefficients, score


def two_class_weighted_score(a, b, predicted_a, predicted_b, kappa):
    """
    Convenience function for computing weighted RMSE in
        energy and force predictions.

    Args:
        a (np.ndarray): energy reference values.
        b (np.ndarray): force reference values.
        predicted_a (np.ndarray): energy predictions.
        predicted_b (np.ndarray): force predictions.
        kappa: weighting parameter between 0 and 1, e.g. 1 ignores forces.

    Returns:
        score (float).
    """
    a_std = np.std(a)
    b_std = np.std(b)
    a_rmse = np.sqrt(np.mean(np.subtract(predicted_a, a) ** 2)) / a_std
    b_rmse = np.sqrt(np.mean(np.subtract(predicted_b, b) ** 2)) / b_std
    score = a_rmse * kappa + (1 - kappa) * b_rmse
    return score


def weighted_cross_validation(x, fx, y, fy, kappa, n_folds=5,
                              lambda1=0, lambda2=0):
    """
    Args:
        x (np.ndarray): Inputs matrix (energy).
        y (list): Outputs vector (energy).
        fx (np.ndarray): Inputs matrix (forces).
        fy (list): Outputs vector (forces).
        kappa (float): weight parameter e.g. 1 to ignore force contribution.
        n_folds: number of folds (k) for k-fold cross validation.
        lambda1 (float): L2 regularization strength (multiplicative)
            for ridge regression.
        lambda2 (float): curvature regularization strength (multiplicative).

    Returns:
        rmse (float): Root-mean-square error from direct evaluation of func.
        cv_rmse (float): Root-mean-square error from k-fold cross validation.
    """
    n_features = x.shape[1]

    reg = regularize.get_regularizer_matrix(n_features,
                                            ridge=lambda1,
                                            curvature=lambda2)

    e_indices = np.arange(len(x))
    f_indices = np.arange(len(fx))
    np.random.shuffle(e_indices)
    np.random.shuffle(f_indices)
    e_folds = np.array_split(e_indices, n_folds)
    f_folds = np.array_split(f_indices, n_folds)

    pt_cv = []
    gt_cv = []
    y_cv = []
    fy_cv = []
    for k in range(n_folds):
        x_holdout = np.take(x, e_folds[k], axis=0)
        y_holdout = np.take(y, e_folds[k], axis=0)
        fx_holdout = np.take(fx, f_folds[k], axis=0)
        fy_holdout = np.take(fy, f_folds[k], axis=0)

        e_fold = np.concatenate(np.delete(e_folds, k, axis=0), axis=0)
        f_fold = np.concatenate(np.delete(f_folds, k, axis=0), axis=0)
        x_fold = np.take(x, e_fold, axis=0)
        y_fold = np.take(y, e_fold, axis=0)
        fx_fold = np.take(fx, f_fold, axis=0)
        fy_fold = np.take(fy, f_fold, axis=0)

        coefficients = weighted_least_squares([x_fold, fx_fold],
                                              [y_fold, fy_fold],
                                              [kappa, 1-kappa],
                                              regularizers=reg)
        pt = np.dot(x_holdout, coefficients)
        gt = np.dot(fx_holdout, coefficients)
        pt_cv.extend(pt)
        gt_cv.extend(gt)
        y_cv.extend(y_holdout)
        fy_cv.extend(fy_holdout)
    cv_score = two_class_weighted_score(y_cv, fy_cv, pt_cv, gt_cv, kappa)
    return cv_score