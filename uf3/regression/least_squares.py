import numpy as np
from uf3.regression import regularize


class WeightedLinearModel:
    """
    Scikit-learn compatible class for weighted linear regression.
    Fit model given x, y, optional weights, and optional regularizer.
    """
    def __init__(self,
                 weights=None,
                 fixed=None,
                 regularizer=None,
                 **regularizer_params):
        """
        Args:
            weights (np.ndarray): sample weights (optional).
            fixed (list): list of tuples of indices and coefficients to fix
                before fitting. Useful for ensuring smooth cutoffs or
                fixing multiplicative coefficients.
                e.g. fix=[(0, 10), (15, 0)] fixes the first coefficient (idx=0)
                to 10 and the sixteenth coefficient (idx=15) to 0.
            regularizer (uf3.regularize.Regularizer): regularization
                handler to query for regularization matrix.
            regularizer_params: arguments to generate regularizer matrix
                if regularizer is not provided.
        """
        if isinstance(weights, (list, np.ndarray)):
            self.weights = weights
        self.coefficients = None
        self.fixed = fixed
        if regularizer is not None:
            self.regularizer = regularizer.matrix
        else:
            if regularizer_params is None:
                raise ValueError(
                    "Neither regularizer nor regularizer parameters provided.")
            regularizer = regularize.Regularizer(**regularizer_params)
            self.regularizer = regularizer.matrix

    def fit(self, x, y):
        """
        Args:
            x (np.ndarray): input matrix of shape (n_samples, n_features).
            y (np.ndarray): output vector of length n_samples.
        """
        solution, _ = weighted_least_squares(x,
                                             y,
                                             self.weights,
                                             self.regularizer,
                                             fixed=self.fixed)
        self.coefficients = solution

    def predict(self, x):
        """
        Args:
            x (np.ndarray): input matrix of shape (n_samples, n_features).

        Returns:
            predictions (np.ndarray): vector of predictions.
        """
        predictions = np.dot(x, self.coefficients)
        return predictions

    def score(self, x, y, weights=None):
        """
        Args:
            x (np.ndarray): input matrix of shape (n_samples, n_features).
            y (np.ndarray): output vector of length n_samples.
            weights (np.ndarray): sample weights (optional).

        Returns:
            score (float): weighted root-mean-square-error of prediction.
        """
        n_features = len(x[0])
        if weights is not None:
            w_matrix = np.eye(n_features) * np.sqrt(weights)
            x = np.dot(w_matrix, x)
            y = np.dot(w_matrix, y)
        predictions = self.predict(x)
        score = np.sqrt(np.mean(np.subtract(y, predictions) ** 2))
        return score


def linear_least_squares(x, y):
    """
    Solves the linear least-squares problem Ax=y. Regularizer matrix
        should be concatenated to x and zero-values padded to y.

    Args:
        x (np.ndarray): input matrix of shape (n_samples, n_features).
        y (np.ndarray): output vector of length n_samples.

    Returns:
        solution (np.ndarray): coefficients.
    """
    xTx = np.dot(x.T, x)
    xTx_inv = np.linalg.inv(xTx)
    solution = np.dot(np.dot(xTx_inv, x.T), y)
    return solution


def weighted_least_squares(x,
                           y,
                           weights=None,
                           regularizers=None,
                           fixed=None):
    """
    Solves the linear least-squares problem with optional square regularizer
        matrix and optional weighting.

    Args:
        x (np.ndarray): input matrix.
        y (np.ndarray): output vector.
        weights (np.ndarray): sample weights (optional).
        regularizers (np.ndarray, list): matrix or list of matrices.
        fixed (list): list of tuples of indices and coefficients to fix
            before fitting. Useful for ensuring smooth cutoffs or
            fixing multiplicative coefficients.
            e.g. fix=[(0, 10), (15, 0)] fixes the first coefficient (idx=0)
            to 10 and the sixteenth coefficient (idx=15) to 0.

    Returns:
        solution (np.ndarray): coefficients.
        predictions (list of np.ndarray): predictions.
    """
    n_feats = len(x[0])
    if regularizers is None:
        regularizers = ()
    elif isinstance(regularizers, np.ndarray):
        if regularizers.shape == (n_feats, n_feats):
            regularizers = [regularizers]
        else:
            raise ValueError(
                "Expected regularizer shape: {} x {}".format(n_feats, n_feats))

    if weights is not None:
        if len(weights) != len(x):
            raise ValueError(
                'Number of weights does not match number of samples.')
        if not np.all(np.positive(weights)):
            raise ValueError('Negative weights provided.')
        w_matrix = np.eye(len(x)) * np.sqrt(weights)
        x_fit = np.dot(w_matrix, x.copy())
        y_fit = np.dot(w_matrix, y.copy())
    else:
        x_fit = x.copy()
        y_fit = y.copy()

    if fixed is not None:
        fixed = np.array(fixed)
        fixed_colidx = fixed[:, 0]
        fixed_coefficients = fixed[:, 1]
        x_fit, y_fit, mask = preprocess_fixed_coefficients(x_fit,
                                                           y_fit,
                                                           regularizers,
                                                           fixed_coefficients,
                                                           fixed_colidx)

    reg_zeros = [np.zeros(len(array)) for array in regularizers]
    x_fit = np.concatenate([x_fit, *regularizers])
    y_fit = np.concatenate([y_fit, *reg_zeros])
    solution = linear_least_squares(x_fit, y_fit)

    if fixed is None:
        return solution
    else:
        coefficients = np.zeros(n_feats)
        np.put_along_axis(coefficients, mask, solution)
        np.put_along_axis(coefficients, fixed_coefficients, fixed_colidx)
        return coefficients


def preprocess_fixed_coefficients(x,
                                  y,
                                  regularizers,
                                  fixed_coefficients,
                                  fixed_colidx):
    """
    
    Args:
        x (np.ndarray): feature array.
        y (np.ndarray): target vector.
        regularizers (list): list of regularization matrices.
        fixed_coefficients (np.ndarray): coefficient values to fix.
        fixed_colidx (np.ndarray): column indices of fixed coefficients.

    Returns:
        x (np.ndarray): feature array with fixed_colidx removed.
        y (np.ndarray): target vector with fixed column contributes subtracted.
        mask (np.ndarray): indices of remaining columns in x.
    """
    n_feats = len(x[0])
    mask = np.setdiff(np.arange(n_feats), fixed_colidx)
    x = x[:, mask]
    A_fixed = x[:, fixed_colidx]
    y = np.subtract(y, np.dot(A_fixed, fixed_coefficients))
    for i, array in enumerate(regularizers):
        regularizers[i] = array[mask, mask]
    return x, y, mask
