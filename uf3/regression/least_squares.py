import numpy as np
from sklearn import model_selection
from uf3.regression import regularize


DEFAULT_REGULARIZER_GRID = dict(ridge_1b=[1e-3],
                                ridge_2b=np.geomspace(1e-6, 1, 7),
                                ridge_3b=[1e-5],
                                curve_2b=np.geomspace(1e-6, 1, 7),
                                curve_3b=[1e-4])


class WeightedLinearModel:
    """
    -Perform weighted linear regression with scikit-learn compatible functions
    -Fit model given x, y, optional weights, and optional regularizer
    """
    def __init__(self,
                 bspline_config=None,
                 regularizer=None,
                 fixed_tuples=None,
                 mask_zeros=False,
                 **params):
        """
        Args:
            bspline_config (bspline.BsplineConfig)
            regularizer (np.ndarray): regularization matrix.
            fixed_tuples (list): list of tuples of (column_index, value)
                to fix certain coefficient values during fit.
            mask_zeros (bool): drop columns with zero variance and zero mean
                during fitting, ensuring zero-valued coefficients.
            params: arguments to generate regularizer matrix.
        """
        self.coefficients = None
        self.fixed_tuples = fixed_tuples
        self.regularizer = regularizer
        self.bspline_config = bspline_config
        self.mask_zeros = mask_zeros
        if self.regularizer is None:
            # initialize regularizer matrix if unspecified.
            self.set_params(**params)

    def set_params(self, **params):
        """Set parameters from keyword arguments. Initializes
            regularizer with default parameters if unspecified."""
        if "bspline_config" in params:
            self.bspline_config = params["bspline_config"]
        if "fixed_tuples" in params:
            self.fixed_tuples = params["fixed_tuples"]
        if "mask_zeros" in params:
            self.mask_zeros = params["mask_zeros"]
        if "regularizer" in params:
            self.regularizer = params["regularizer"]
        elif self.regularizer is None:
            params = {k: v for k, v in params.items()
                      if k in DEFAULT_REGULARIZER_GRID}
            reg = self.bspline_config.get_regularization_matrix(**params)
            self.regularizer = reg

    def fit(self, x, y, weights=None):
        """
        Args:
            x (np.ndarray): input matrix of shape (n_samples, n_features).
            y (np.ndarray): output vector of length n_samples.
            weights (np.ndarray): sample weights (optional).
        """
        _, n_features = x.shape
        if self.mask_zeros:
            var = np.var(x, axis=0) != 0
            m = np.mean(x, axis=0) != 0
            mask = np.where((var & m))[0]
            x = x[:, mask]
            regularizer = self.regularizer[mask[:, None], mask[None, :]]
            if self.fixed_tuples is not None:
                fixed_tuples = [v for v in self.fixed_tuples
                                if v[0] in mask]
            else:
                fixed_tuples = self.fixed_tuples
            print(x.shape, regularizer.shape, np.sum(mask))
            solution = weighted_least_squares(x,
                                              y,
                                              weights=weights,
                                              regularizer=regularizer,
                                              fixed=fixed_tuples)
            padded_solution = np.zeros(n_features)
            padded_solution[mask] = solution
            self.coefficients = padded_solution
        else:
            solution = weighted_least_squares(x,
                                              y,
                                              weights=weights,
                                              regularizer=self.regularizer,
                                              fixed=self.fixed_tuples)
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
            score (float): negative weighted root-mean-square-error.
        """
        n_features = len(x[0])
        if weights is not None:
            w_matrix = np.eye(n_features) * np.sqrt(weights)
            x = np.dot(w_matrix, x)
            y = np.dot(w_matrix, y)
        predictions = self.predict(x)
        score = -np.sqrt(np.mean(np.subtract(y, predictions) ** 2))
        return score

    @staticmethod
    def optimize(x,
                 y,
                 bspline_config,
                 fixed_tuples=None,
                 mask_zeros=False,
                 weights=None,
                 factor=3,
                 grid_points=5,
                 outer_cv=5,
                 inner_cv=10,
                 seed=0,
                 n_jobs=-1,
                 verbose=False,
                 **regularizer_grids):
        regularizer_space = dict(DEFAULT_REGULARIZER_GRID)
        for k, v in regularizer_grids.items():
            if isinstance(v, tuple):
                if len(v) == 2:
                    space = np.linspace(v[0], v[1], grid_points)
                elif len(v) > 2:
                    space = np.linspace(v[0], v[1], v[2])
                else:
                    print("Incorrect search space defined for {}.".format(k))
                    continue
            else:
                space = [v]
            regularizer_space[k] = space
        cv_inner = model_selection.KFold(n_splits=inner_cv,
                                         shuffle=True,
                                         random_state=seed)
        cv_outer = model_selection.KFold(n_splits=outer_cv,
                                         shuffle=True,
                                         random_state=seed)
        model = WeightedLinearModel(bspline_config=bspline_config,
                                    fixed_tuples=fixed_tuples,
                                    mask_zeros=mask_zeros)
        search = model_selection.HalvingGridSearchCV(model,
                                                     regularizer_space,
                                                     factor=factor,
                                                     resource='n_samples',
                                                     cv=cv_inner,
                                                     n_jobs=1,
                                                     refit=True)
        if weights is not None:
            fit_params = {"weights": weights}
        else:
            fit_params = None
        scores = model_selection.cross_validate(search,
                                                x,
                                                y,
                                                cv=cv_outer,
                                                return_estimator=True,
                                                verbose=verbose,
                                                n_jobs=n_jobs,
                                                fit_params=fit_params)
        optimized_params = [estimator.best_params_ for estimator in
                            scores['estimator']]
        model.set_params()


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
                           regularizer=None,
                           fixed=None):
    """
    Solves the linear least-squares problem with optional square regularizer
        matrix and optional weighting.

    Args:
        x (np.ndarray): input matrix.
        y (np.ndarray): output vector.
        weights (np.ndarray): sample weights (optional).
        regularizer (np.ndarray)
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
    if regularizer is not None:
        if regularizer.shape != (n_feats, n_feats):
            s1, s2 = regularizer.shape
            shape_comparison = "{0} x {0}. Provided: {1} x {2}".format(n_feats,
                                                                       s1,
                                                                       s2)
            raise ValueError(
                "Expected regularizer shape: " + shape_comparison)

    if weights is not None:
        if len(weights) != len(x):
            raise ValueError(
                'Number of weights does not match number of samples.')
        if not np.all(weights >= 0):
            raise ValueError('Negative weights provided.')
        w = np.sqrt(weights)
        x_fit = np.multiply(x.copy().T, w).T
        y_fit = np.multiply(y.copy(), w)
    else:
        x_fit = x.copy()  # copy in preparation for modifying with fixed coeff.
        y_fit = y.copy()

    if fixed is not None:
        fixed = np.array(fixed)
        fixed_colidx = fixed[:, 0].astype(int)
        fixed_coefficients = fixed[:, 1]
        x_fit, y_fit, mask = preprocess_fixed_coefficients(x_fit,
                                                           y_fit,
                                                           fixed_coefficients,
                                                           fixed_colidx)
        regularizer = regularizer[mask, :][:, mask]

    if regularizer is not None:
        reg_zeros = np.zeros(len(regularizer))
        x_fit = np.concatenate([x_fit, regularizer])
        y_fit = np.concatenate([y_fit, reg_zeros])
    solution = linear_least_squares(x_fit, y_fit)

    if fixed is None:
        return solution
    else:
        coefficients = np.zeros(n_feats)
        np.put_along_axis(coefficients, mask, solution, 0)
        np.put_along_axis(coefficients, fixed_colidx, fixed_coefficients, 0)
        return coefficients


def preprocess_fixed_coefficients(x,
                                  y,
                                  fixed_coefficients,
                                  fixed_colidx):
    """
    
    Args:
        x (np.ndarray): feature array.
        y (np.ndarray): target vector.
        fixed_coefficients (np.ndarray): coefficient values to fix.
        fixed_colidx (np.ndarray): column indices of fixed coefficients.

    Returns:
        x (np.ndarray): feature array with fixed_colidx removed.
        y (np.ndarray): target vector with fixed column contributes subtracted.
        mask (np.ndarray): indices of remaining columns in x.
    """
    n_feats = len(x[0])
    mask = np.setdiff1d(np.arange(n_feats), fixed_colidx)
    x_fixed = x[:, fixed_colidx]
    x = x[:, mask]
    y = np.subtract(y, np.dot(x_fixed, fixed_coefficients))
    return x, y, mask

