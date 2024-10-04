"""
This module provides the WeightedLinearModel class for fitting UF potentials
from featurized DataFrames using regularized least squares.
"""

from typing import List, Dict, Collection, Tuple
import os
import warnings
import numpy as np
import pandas as pd
import ndsplines
from uf3.representation import bspline, process
from uf3.data import io
from uf3.data import composition
from uf3.util import json_io
from uf3.util import parallel


class VarianceRecorder:
    """Convenience class for computing online variance and mean"""
    def __init__(self, mean=0, std=0, n=0):
        self.mean = mean
        self.std = std
        self.n = int(n)

    def update(self, batch: Collection) -> Tuple:
        """
        Args:
            batch (list or np.ndarray): n-dimensional data. For speed purposes,
                dimensions are not checked for compatibility so caution
                is advised when working with multidimensional data.
                Statistics are computed along the first axis.

        Returns:
            (current mean, current standard deviation, current entry count)
        """
        if self.n == 0:
            self.mean = np.mean(batch, axis=0)
            self.std = np.std(batch, axis=0)
            self.n = len(batch)
            return self.mean, self.std, self.n
        else:
            batch_std = np.std(batch, axis=0)
            batch_mean = np.mean(batch, axis=0)
            m = float(self.n)
            n = len(batch)
            std = (m / (m + n) * self.std**2
                   + n / (m + n) * batch_std**2
                   + m * n / (m + n)**2 * (self.mean - batch_mean)**2)
            self.std = np.sqrt(std)
            self.mean = m / (m + n) * self.mean + n / (m + n) * batch_mean
            self.n += n
            return self.mean, self.std, self.n

    def update_with_components(self, df, keys=None):
        """Wrapper for dataframe with multiple columns of interest"""
        if keys is None:
            keys = ["fx", "fy", "fz"]
        batch = []
        for j, *components in df[keys].itertuples():
            if any([component is np.nan for component in components]):
                continue
            if np.ndim(components) > 1:  # if components are not scalars
                components = list(np.concatenate(components))
            batch.extend(components)
        self.update(batch)
        return self.mean, self.std, self.n


class BasicLinearModel:
    """
    Base class for linear regression.
    """
    def __init__(self,
                 regularizer: np.ndarray = None):
        """
        Args:
            regularizer (np.ndarray): regularization matrix.
        """
        self.coefficients = None
        self.regularizer = regularizer

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            ridge_penalty: float = 1e-8,
            ):
        """
        Direct solution to linear least squares with LU decomposition.

        Args:
            x (np.ndarray): input matrix of shape (n_samples, n_features).
            y (np.ndarray): output vector of length n_samples.
            ridge_penalty (float): magnitude of ridge penalty. Ignored
                if self.regularizer is set at initialization.
        """
        gram, ordinate = moore_penrose_components(x, y)
        if self.regularizer is None:
            regularizer = np.eye(len(gram)) * ridge_penalty
        else:
            regularizer = self.regularizer
        regularizer = np.dot(regularizer.T, regularizer)
        coefficients = lu_factorization(gram + regularizer, ordinate)
        self.coefficients = coefficients

    def predict(self, x: np.ndarray):
        """
        Predict using fit coefficients.

        Args:
            x (np.ndarray): input matrix of shape (n_samples, n_features).

        Returns:
            predictions (np.ndarray): vector of predictions.
        """
        predictions = np.dot(x, self.coefficients)
        return predictions

    def score(self, x, y, weights=None, normalize=True):
        """
        Evaluate score (negative error metric).

        Args:
            x (np.ndarray): input matrix of shape (n_samples, n_features).
            y (np.ndarray): output vector of length n_samples.
            weights (np.ndarray): sample weights (optional).
            normalize (bool): whether to normalize by the std of y.

        Returns:
            score (float): negative weighted root-mean-square-error.
        """
        n_features = len(x[0])
        if weights is not None:
            w_matrix = np.eye(n_features) * np.sqrt(weights)
            x = np.dot(w_matrix, x)
            y = np.dot(w_matrix, y)
        predictions = self.predict(x)
        score = -rmse_metric(y, predictions)
        if normalize:
            score /= np.std(y)
        return score


class WeightedLinearModel(BasicLinearModel):
    """
    Handler class for regularized linear least squares using energies and
    forces and basis set provided by bspline.BsplineBasis.
    """
    def __init__(self,
                 bspline_config,
                 regularizer=None,
                 data_coverage=None,
                 **params):
        super().__init__(regularizer)
        self.bspline_config = bspline_config
        n_basis = np.sum(self.bspline_config.get_feature_partition_sizes())
        if data_coverage is not None:
            if len(data_coverage) == n_basis:
                self.data_coverage = data_coverage
            else:
                raise ValueError(
                    f"Incorrect data_coverage shape: "
                    f"{len(data_coverage)} != {n_basis}"
                )
        else:
            self.data_coverage = np.zeros(n_basis, dtype=bool)

        if self.regularizer is None:
            # initialize regularizer matrix if unspecified.
            self.set_params(**params)

    def set_params(self, **params):
        """Set parameters from keyword arguments. Initializes
            regularizer with default parameters if unspecified."""
        if "bspline_config" in params:
            self.bspline_config = params["bspline_config"]
        if "regularizer" in params:
            self.regularizer = params["regularizer"]
        elif self.regularizer is None:
            reg_params = {k: v for k, v in params.items()
                          if isinstance(v, (int, float, np.floating))}
            reg = self.bspline_config.get_regularization_matrix(**reg_params)
            self.regularizer = reg

    @staticmethod
    def from_config(config):
        return WeightedLinearModel.from_dict(config)

    @staticmethod
    def from_dict(config):
        bspline_config = bspline.BSplineBasis.from_dict(config)
        regularizer = config.get("regularizer", None)
        data_coverage = config.get("data_coverage", None)
        model = WeightedLinearModel(bspline_config,
                                    regularizer=regularizer,
                                    data_coverage=data_coverage)
        model.load(solution=config)
        return model

    @staticmethod
    def from_json(filename):
        """Load model (coefficients and knots map) from json file."""
        dump = json_io.load_interaction_map(filename)
        return WeightedLinearModel.from_dict(dump)

    def as_dict(self):
        solution = arrange_coefficients(self.coefficients, self.bspline_config)
        for trio in self.bspline_config.interactions_map.get(3, []):
            solution[trio] = self.bspline_config.decompress_3B(solution[trio],
                                                               trio)
        knots_map = self.bspline_config.knots_map
        dump = dict(coefficients=solution,
                    knots=knots_map,
                    data_coverage=self.data_coverage,
                    **self.bspline_config.as_dict())
        return dump

    @property
    def n_feats(self):
        return self.bspline_config.n_feats

    @property
    def frozen_c(self):
        return self.bspline_config.frozen_c

    @property
    def col_idx(self):
        return self.bspline_config.col_idx

    @property
    def mask(self):
        return get_freezing_mask(self.n_feats, self.col_idx)

    def __repr__(self):
        if self.coefficients is None:
            fit = "False"
        else:
            fit = "True"
        summary = ["WeightedLinearModel:",
                   f"    Fit: {fit}",
                   self.bspline_config.__repr__()
                   ]
        return "\n".join(summary)

    def __str__(self):
        return self.__repr__()

    def fit_with_gram(self, gram: np.ndarray, ordinate: np.ndarray):
        """
        Intermediate function for direct solution using gram matrix
        and ordinate (Moore-penrose inverse).

        Args:
            gram (np.ndarray): gram matrix (x^T x)
            ordinate (np.ndarray: ordinate (x^T y)
        """
        data_coverage = (np.sum(gram, axis=0) != 0)
        data_coverage = revert_frozen_coefficients(data_coverage,
                                                   self.n_feats,
                                                   self.mask,
                                                   self.frozen_c,
                                                   self.col_idx)
        self.data_coverage = np.logical_or(self.data_coverage, data_coverage)
        regularizer = freeze_regularizer(self.regularizer, self.mask)
        regularizer = np.dot(regularizer.T, regularizer)
        coefficients = lu_factorization(gram + regularizer, ordinate)
        coefficients = revert_frozen_coefficients(coefficients,
                                                  self.n_feats,
                                                  self.mask,
                                                  self.frozen_c,
                                                  self.col_idx)
        self.coefficients = coefficients

    def fit(self,
            x_e: np.ndarray,
            y_e: np.ndarray,
            x_f: np.ndarray = None,
            y_f: np.ndarray = None,
            weight: float = 0.5,
            batch_size=2500,
            ):
        """
        Direct solution from input-output pairs corresponding to
        energies and forces, with option to weigh their respective
        contributions.

        Args:
            x_e (np.ndarray): input matrix of shape (n_samples, n_features).
            y_e (np.ndarray): output vector of length n_samples.
            x_f (np.ndarray): input matrix corresponding to forces.
            y_f (np.ndarray): output vector corresponding to forces.
            weight (float): parameter balancing contribution from energies
                vs. forces. Higher values favor energies; defaults to 0.5.
            batch_size: maximum batch size for gram matrix construction.
        """
        x_e, y_e = freeze_columns(x_e,
                                  y_e,
                                  self.mask,
                                  self.frozen_c,
                                  self.col_idx)
        gram_e, ord_e = batched_moore_penrose(x_e, y_e, batch_size=batch_size)
        if x_f is not None:
            energy_weight, force_weight = calc_E_F_weights(len(y_e),
                                                           len(y_f),
                                                           np.std(y_e),
                                                           np.std(y_f))
            x_f, y_f = freeze_columns(x_f,
                                      y_f,
                                      self.mask,
                                      self.frozen_c,
                                      self.col_idx)
            gram_f, ord_f = batched_moore_penrose(x_f,
                                                  y_f,
                                                  batch_size=batch_size)
            gram, ordinate = self.combine_weighted_gram(gram_e, gram_f, ord_e,
                                                        ord_f, energy_weight,
                                                        force_weight, weight)
        else:
            gram = gram_e
            ordinate = ord_e
        self.fit_with_gram(gram, ordinate)

    def combine_weighted_gram(self,
                              gram_e: np.ndarray,
                              gram_f: np.ndarray,
                              ord_e: np.ndarray,
                              ord_f: np.ndarray,
                              energy_weight: float,
                              force_weight: float,
                              weight: float):
        """
        Apply weighting to gram matrices and ordinates for energy and
        force contributions to the fit.

        Args:
            gram_e (np.ndarray): gram matrix (x^T x) for energies.
            gram_f (np.ndarray): gram matrix (x^T x) for forces.
            ord_e (np.ndarray): ordinate (x^T y) for energies.
            ord_f (np.ndarray): ordinate (x^T y) for forces.
            energy_weight: 1 / (# energies * sqrt(Var(energies)))
            force_weight: 1 / (# forces * sqrt(Var(forces)))
            weight (float): parameter balancing contribution from energies
                vs. forces. Higher values favor energies; defaults to 0.5.

        Returns:
            gram (np.ndarray): gram matrix (x^T x) for fitting.
            ordinate (np.ndarray): ordinate (x^T y) for fitting.
        """
        gram = ((weight * energy_weight**2 * gram_e)
                + ((1 - weight) * force_weight**2 * gram_f))
        ordinate = ((weight * energy_weight**2 * ord_e)
                    + ((1 - weight) * force_weight**2 * ord_f))
        return gram, ordinate

    def fit_from_file(self,
                      filename: str,
                      subset: Collection,
                      weight: float = 0.5,
                      batch_size=2500,
                      sample_weights: Dict = None,
                      energy_key="energy",
                      progress: str = "bar",
                      drop_columns: List[str] = None):
        """
        Accumulate inputs and outputs from batched parsing of HDF5 file
        and compute direct solution via LU decomposition.

        Args:
            filename (str): path to HDF5 file.
            subset (list): list of keys for training.
            weight (float): parameter balancing contribution from energies
                vs. forces. Higher values favor energies; defaults to 0.5.
            batch_size (int): batch size, in rows, for matrix multiplication
                operations in constructing gram matrices.
            sample_weights (dict):
            energy_key (str): column name for energies, default "energy".
            progress (str): style for progress indicators.
            drop_columns (list): list of columns to drop. Used when modifying
                the cutoffs of the feature vectors from HDF5 file. No internal
                checks are performed to see if dropping provided columns produce
                features of the intended cutoffs. Use with Caution.
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        n_tables, _, table_names, _ = io.analyze_hdf_tables(filename)
        gram_e, gram_f, ord_e, ord_f = self.initialize_gram_ordinate()
        e_variance = VarianceRecorder()
        f_variance = VarianceRecorder()
        table_iterator = parallel.progress_iter(np.arange(n_tables),
                                                style=progress)
        for j in table_iterator:
            table_name = table_names[j]
            df = process.load_feature_db(filename, table_name)
            keys = df.index.unique(level=0).intersection(subset)
            if len(keys) == 0:
                continue

            if drop_columns != None:
                df.drop(columns=drop_columns,inplace=True)

            intermediates = self.gram_from_df(df,
                                              keys,
                                              e_variance=e_variance,
                                              f_variance=f_variance,
                                              sample_weights=sample_weights,
                                              energy_key=energy_key,
                                              batch_size=batch_size)
            g_e, g_f, o_e, o_f = intermediates
            gram_e += g_e
            gram_f += g_f
            ord_e += o_e
            ord_f += o_f
        energy_weight, force_weight = calc_E_F_weights(e_variance.n,
                                                       f_variance.n,
                                                       e_variance.std,
                                                       f_variance.std)
        gram, ordinate = self.combine_weighted_gram(gram_e,
                                                    gram_f,
                                                    ord_e,
                                                    ord_f,
                                                    energy_weight,
                                                    force_weight,
                                                    weight)
        self.fit_with_gram(gram, ordinate)

    def initialize_gram_ordinate(self):
        """Initialize empty matrices for gram matrices and ordinates."""
        n_columns = self.n_feats - len(self.col_idx)
        gram_e = np.zeros((n_columns, n_columns))
        ord_e = np.zeros(n_columns)
        gram_f = np.zeros((n_columns, n_columns))
        ord_f = np.zeros(n_columns)
        return gram_e, gram_f, ord_e, ord_f

    def gram_from_df(self,
                     df: pd.DataFrame,
                     keys: Collection,
                     e_variance: VarianceRecorder = None,
                     f_variance: VarianceRecorder = None,
                     sample_weights: Dict = None,
                     energy_key: str = "energy",
                     batch_size: int = 2500):
        """
        Extract inputs and outputs from dataframe and compute
        moore-penrose components (gram matrices and ordinates).

        Args:
            df (pd.DataFrame): DataFrame of energy/force features.
            keys (list): keys to query from df (e.g. training subset).
            e_variance (VarianceRecorder): handler for accumulating
                statistics for energies (mean and variance).
            f_variance (VarianceRecorder): handler for accumulating
                statistics for forces (mean and variance).
            sample_weights (dict):
            energy_key (str): column name for energies, default "energy".
            batch_size (int): batch size, in rows, for matrix multiplication
                operations in constructing gram matrices.
        """
        n_elements = len(self.bspline_config.element_list)
        x_e, y_e, x_f, y_f = dataframe_to_tuples(df.loc[keys],
                                                 n_elements=n_elements,
                                                 energy_key=energy_key,
                                                 sample_weights=sample_weights)
        x_e, y_e = freeze_columns(x_e,
                                  y_e,
                                  self.mask,
                                  self.frozen_c,
                                  self.col_idx)
        x_f, y_f = freeze_columns(x_f,
                                  y_f,
                                  self.mask,
                                  self.frozen_c,
                                  self.col_idx)
        if e_variance is not None and f_variance is not None:
            e_variance.update(y_e)
            f_variance.update(y_f)
        gram_e, ordinate_e = batched_moore_penrose(x_e,
                                                   y_e,
                                                   batch_size=batch_size)
        gram_f, ordinate_f = batched_moore_penrose(x_f,
                                                   y_f,
                                                   batch_size=batch_size)
        return gram_e, gram_f, ordinate_e, ordinate_f

    def batched_predict(self,
                        filename: str,
                        keys: List[str] = None,
                        table_names: List[str] = None,
                        score: bool = True,
                        drop_columns: List[str] = None):
        """
        Extract inputs and outputs from HDF5 file and predict energies/forces.

        Args:
            filename: path to HDF5 file.
            keys (list): keys to query from df (e.g. training subset).
            table_names (list): list of table names in HDF5 to read.
            score (bool): whether to return root mean square error metrics.

        Returns:
            y_e (np.ndarray): target values for energies.
            p_e (np.ndarray): prediction values for energies.
            y_f (np.ndarray): target values for forces.
            p_f (np.ndarray): prediction values for forces.
            rmse_e (np.ndarray): RMSE across energy predictions.
            rmse_e (np.ndarray): RMSE across force predictions.
            drop_columns (list): list of columns to drop. Used when modifying
                the cutoffs of the feature vectors from HDF5 file. No internal
                checks are performed to see if dropping provided columns produce
                features of the intended cutoffs. Use with Caution.
        """
        n_elements = len(self.bspline_config.element_list)
        y_e, p_e, y_f, p_f = batched_prediction(self,
                                                filename,
                                                table_names=table_names,
                                                subset_keys=keys,
                                                n_elements=n_elements,
                                                drop_columns=drop_columns)
        if score:
            rmse_e = rmse_metric(y_e, p_e)
            rmse_f = rmse_metric(y_f, p_f)
            print(f"RMSE (energy): {rmse_e:.3F}")
            print(f"RMSE (forces): {rmse_f:.3F}")
            return y_e, p_e, y_f, p_f, rmse_e, rmse_f
        else:
            return y_e, p_e, y_f, p_f

    def to_json(self, filename: str):
        """Save model (coefficients and knots map) to json file."""
        json_io.dump_interaction_map(self.as_dict(),
                                     filename=filename,
                                     write=True)

    def dump(self):
        """Legacy alias"""
        return self.as_dict()

    def load(self,
             solution: Dict = None,
             filename: str = None,
             ):
        """
        Reflatten coefficients (e.g. obtained through arrange_coefficients)
        and load into model for prediction.

        Args:
            solution (dict): dictionary of 1B, 2B, ... terms
                organized as interaction: vector entries.
            filename (str): filename of json dump containing solution.
        """
        if filename is not None:
            if solution is not None:
                warnings.warn("Provided solutions ignored; loading file.")
            solution = json_io.load_interaction_map(filename)
        elif solution is None:
            raise ValueError("Neither solution nor filename were provided.")
        if "coefficients" in solution:
            solution = solution["coefficients"]
        elif "solution" in solution:
            # TODO: proper deprecation
            warnings.warn("'solution' should be renamed to 'coefficients'")
            solution = solution["solution"]
        for key in solution:
            if isinstance(key, tuple):
                sorted_key = composition.sort_interaction_symbols(key)
                if sorted_key != key:
                    solution[sorted_key] = solution[key]
        # consistency check with bspline_config
        component_len = self.bspline_config.get_interaction_partitions()[0]
        for pair in self.bspline_config.interactions_map[2]:
            n_target = component_len[pair]
            if pair not in solution:
                warnings.warn(f"{pair} not provided.")
                solution[pair] = np.zeros(n_target)
            n_provided = len(solution[pair])
            if n_provided != n_target:
                raise ValueError(
                    f"Incorrect shape: {pair}, {n_provided} != {n_target}"
                )
        for trio in self.bspline_config.interactions_map.get(3, []):
            n_target = component_len[trio]
            if trio not in solution:
                warnings.warn(f"{trio} not provided.")
            if trio in solution:
                # decompress if necessary
                component = np.array(solution[trio])
                if len(np.shape(component)) > 1:
                    vector = self.bspline_config.compress_3B(component,
                                                             trio,
                                                             fitting = False)
                    solution[trio] = vector
            n_provided = len(solution[trio])
            if n_provided != n_target:
                raise ValueError(
                    f"Incorrect shape: {trio}, {n_provided} != {n_target}"
                )
        flattened_coefficients = []
        for element in self.bspline_config.element_list:
            value = solution[element]
            flattened_coefficients.append([value])
        for degree in range(2, self.bspline_config.degree + 1):
            interactions = self.bspline_config.interactions_map[degree]
            for interaction in interactions:
                values = solution[interaction]
                flattened_coefficients.append(values)
        # self-energies, pair interactions & trio interactions
        n_interactions = len(self.bspline_config.partition_sizes)
        # add self-energy as separate interactions
        n_coefficients = sum(self.bspline_config.partition_sizes)
        if len(flattened_coefficients) != n_interactions:
            error_line = "Incorrect interactions: {} provided, {} expected."
            error_line = error_line.format(len(flattened_coefficients),
                                           n_interactions)
            raise ValueError(error_line)
        flattened_coefficients = np.concatenate(flattened_coefficients)
        if len(flattened_coefficients) != n_coefficients:
            error_line = "Incorrect coefficients: {} provided, {} expected."
            error_line = error_line.format(len(flattened_coefficients),
                                           n_coefficients)
            raise ValueError(error_line)
        self.coefficients = np.array(flattened_coefficients)

    def fix_repulsion_2b(self, pair, r_target=None, min_curvature=2.0):
        components = self.bspline_config.get_interaction_partitions()
        component_sizes, component_offsets = components
        offset = component_offsets[pair]
        n_basis = component_sizes[pair]
        idx_subset = np.arange(offset, offset + n_basis)
        c_subset = self.coefficients[idx_subset]
        coverage = self.data_coverage[idx_subset]
        min_coverage = np.argmax(coverage == True)
        if min_coverage == 0:
            print(f"Coverage is sufficient; no fix applied to {pair}.")
        idx_fix = np.arange(self.bspline_config.leading_trim[2], min_coverage)

        knot_sequence = self.bspline_config.knots_map[pair]
        r_centers = knot_sequence[2: n_basis + 2]
        if r_target is None:
            r_target = r_centers[min_coverage]
        r_centers = r_centers[idx_fix]
        c_new = get_spline_taylor_expansion(r_target,
                                            r_centers,
                                            c_subset,
                                            knot_sequence,
                                            min_curvature=min_curvature)
        print(f"{pair} Correction: adjusted {len(idx_fix)} coefficients.")
        self.coefficients[idx_subset[idx_fix]] = c_new


def get_spline_taylor_expansion(r_target,
                                r,
                                coefficients,
                                knot_sequence,
                                min_curvature=0.0):
    nd3 = ndsplines.NDSpline([knot_sequence], coefficients, 3)
    y_trace = nd3(r_target, nus=0)
    d1_trace = nd3(r_target, nus=1)
    d2_trace = nd3(r_target, nus=2)
    if min_curvature is not None:
        d2_trace = max(d2_trace, min_curvature)
    dr = r - r_target
    y = y_trace + (d1_trace * dr) + (0.5 * d2_trace * dr ** 2)
    return y


def dataframe_to_tuples(df_features,
                        n_elements=None,
                        energy_key='energy',
                        sample_weights=None):
    """
    Extract energy/force inputs/outputs from DataFrame.

    Args:
        df_features (pd.DataFrame): dataframe with target vector (y) as the
            first column and feature vectors (x) as remaining columns.
        n_elements (int): number of leading columns to consider for size
            normalization.
        energy_key (str): key for energy samples, used to slice df_features
            into energies and forces for weight generation.
        sample_weights (dict):

    Returns:
        x (np.ndarray): features for machine learning.
        y (np.ndarray): target vector.
        w (np.ndarray): weight vector for machine learning.
    """
    names = df_features.index.get_level_values(0)
    y_index = df_features.index.get_level_values(-1)
    energy_mask = (y_index == energy_key)
    force_mask = np.logical_not(energy_mask)
    data = df_features.to_numpy()
    y = data[:, 0]
    x = data[:, 1:]
    y_e = y[energy_mask]
    y_f = y[force_mask]

    if n_elements is not None:
        s = np.sum(x[energy_mask, :n_elements], axis=1)
        x_e = np.divide(x[energy_mask].T, s).T
        y_e = y_e / s
    else:
        x_e = x[energy_mask]
    x_f = x[force_mask]

    if sample_weights is not None:
        w = np.array([sample_weights.get(name, 1.0) for name in names])
        w_e = w[energy_mask]
        w_f = w[force_mask]
        x_e = np.multiply(x_e.T, w_e).T
        y_e = np.multiply(y_e, w_e)
        x_f = np.multiply(x_f.T, w_f).T
        y_f = np.multiply(y_f, w_f)
    return x_e, y_e, x_f, y_f


def moore_penrose_components(x, y):
    """
    Compute gram matrix (x^T x) and ordinate (x^T y).

    Args:
        x (np.ndarray): input matrix of shape (n_samples, n_features).
        y (np.ndarray): output vector of length n_samples.

    Returns:
        a: Gram matrix (X'X)
        b: ordinate (X'y)
    """
    a = np.dot(x.T, x)
    b = np.dot(x.T, y)
    return a, b


def batched_moore_penrose(x, y, batch_size=2500):
    """
    Batched evaluation of gram matrix (x^T x) and ordinate (x^T y).

    Args:
        x (np.ndarray): input matrix of shape (n_samples, n_features).
        y (np.ndarray): output vector of length n_samples.
        batch_size: maximum batch size, default 2500 rows. This option
            should be adjusted based on efficiency/memory tradeoffs.

    Returns:
        a: Gram matrix (X'X)
        b: ordinate (X'y)
    """

    n_samples, n_features = np.shape(x)
    n_batches = int(n_samples / batch_size)
    if n_batches <= 1:
        return moore_penrose_components(x, y)
    else:
        batched_idx = np.array_split(np.arange(len(y)), n_batches)
        gram = np.zeros((n_features, n_features))
        ordinate = np.zeros(n_features)
        for j, batch in enumerate(batched_idx):
            x_x, x_y = moore_penrose_components(x[batch], y[batch])
            gram += x_x
            ordinate += x_y
        return gram, ordinate


def lu_factorization(a, b):
    """
    LU factorization for least-squares solution using np.linalg.solve().

    Args:
        a: coefficients (X) or Gram matrix (X'X)
        b: ordinate (X'y)
    """
    return np.linalg.solve(a, b)


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
    a, b = moore_penrose_components(x, y)
    return lu_factorization(a, b)


def weighted_least_squares(x, y, weights=None, regularizer=None):
    """
    Solves the linear least-squares problem with optional Tikhonov regularizer
    matrix and optional weighting.
    TODO: Remove (deprecated)

    Args:
        x (np.ndarray): input matrix.
        y (np.ndarray): output vector.
        weights (np.ndarray): sample weights (optional).
        regularizer (np.ndarray): Tikhonov regularizer matrix.

    Returns:
        solution (np.ndarray): coefficients.
        predictions (list of np.ndarray): predictions.
    """
    x_fit, y_fit = apply_weights(x, y, weights)
    n_feats = len(x[0])
    if regularizer is not None:  # append regularizer
        # validate_regularizer(regularizer, n_feats)
        reg_zeros = np.zeros(len(regularizer))
        x_fit = np.concatenate([x_fit, regularizer])
        y_fit = np.concatenate([y_fit, reg_zeros])
    solution = linear_least_squares(x_fit, y_fit)
    return solution


def get_freezing_mask(n_feats: int, col_idx: np.ndarray) -> np.ndarray:
    """
    Freezing mask is the set difference between the range of feature indices
    and the indices to be excluded (col_idx).

    Args:
        n_feats (int): number of features.
        col_idx (list): list of indices to be masked.

    Returns:
        mask (np.ndarray): set of non-frozen indices.
    """
    mask = np.setdiff1d(np.arange(n_feats), col_idx)
    return mask


def freeze_columns(x: np.ndarray,
                   y: np.ndarray,
                   mask: np.ndarray,
                   frozen_c: np.ndarray,
                   col_idx: np.ndarray,
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Freeze coefficients of the solution (e.g. forcing to zero) by
    simultaneously eliminating columns of the input and their assumed
    contribution to the output.

    Args:
        x (np.ndarray): input matrix.
        y (np.ndarray): output vector.
        mask (np.ndarray): set of non-frozen indices.
        frozen_c (np.ndarray): values of coefficients to be frozen.
        col_idx (np.ndarray): indices of coefficients to be frozen.

    Returns:
        x (np.ndarray): input matrix without frozen columns.
        y (np.ndarray): output vector, minus frozen contributions.
    """
    x_fixed = x[:, col_idx]
    x = x[:, mask]
    y = np.subtract(y, np.dot(x_fixed, frozen_c))
    return x, y


def freeze_regularizer(regularizer: np.ndarray,
                       mask: np.ndarray) -> np.ndarray:
    """Apply freezing mask to regularizer, eliminating masked columns."""
    regularizer = regularizer[:, mask]
    return regularizer


def revert_frozen_coefficients(solution: np.ndarray,
                               n_coeff: int,
                               mask: Collection[bool],
                               frozen_c: Collection[float],
                               frozen_idx: Collection[int],) -> np.ndarray:
    """
    Reverse freezing operations by arranging learned coefficients
    and frozen coefficients using the mask.

    Args:
        solution: learned solution, excluding frozen coefficients
        n_coeff: number of columns in full (unfrozen) solution
        mask: indices of remaining columns in x.
        frozen_idx: column indices of fixed coefficients.
        frozen_c: frozen coefficients.

    Returns:
        full_solution (np.ndarray)
    """
    full_solution = np.zeros(n_coeff)
    np.put_along_axis(full_solution, mask, solution, 0)
    np.put_along_axis(full_solution, frozen_idx, frozen_c, 0)
    return full_solution


def apply_weighted_gram(gram_matrix: np.ndarray,
                        weight: float) -> np.ndarray:
    """Deprecated utility function for weighting gram matrix."""
    return gram_matrix * weight**2


def apply_weights(x, y, weights):
    """Deprecated utility function for weighting inputs/outputs."""
    if weights is not None:
        if len(weights) != len(x):
            raise ValueError(
                'Number of weights does not match number of samples.')
        if not np.all(weights >= 0):
            raise ValueError('Negative weights provided.')
        w = np.sqrt(weights)
        x_fit = np.multiply(x.T, w).T
        y_fit = np.multiply(y, w)
    else:
        x_fit = x
        y_fit = y
    return x_fit, y_fit


def validate_regularizer(regularizer: np.ndarray, n_feats: int):
    """
    Check for consistency between regularizer matrix and number of features.

    Args:
        regularizer (np.ndarray): regularizer matrix.
        n_feats (int): number of features.
    """
    n_row, n_col = regularizer.shape
    if n_col != n_feats:
        shape_comparison = "N x {0}. Provided: {1} x {2}".format(n_feats,
                                                                 n_row,
                                                                 n_col)
        raise ValueError(
            "Expected regularizer shape: " + shape_comparison)


def subset_prediction(df: pd.DataFrame,
                      model: WeightedLinearModel,
                      subset_keys: Collection = None,
                      **kwargs
                      ) -> Tuple:
    """
    Convenience function for optimization workflow. Read inputs/outputs
    from DataFrame and predict using fitted model.

    Args:
        df (pd.DataFrame): DataFrame of inputs/outputs.
        model (WeightedLinearModel): fitted model.
        subset_keys (list): list of keys to query from DataFrame.

    Returns:
        y_e (np.ndarray): target values for energies.
        p_e (np.ndarray): prediction values for energies.
        y_f (np.ndarray): target values for forces.
        p_f (np.ndarray): prediction values for forces.
    """
    if subset_keys is not None:
        idx = df.index.unique(level=0).intersection(subset_keys)
        if len(idx) == 0:
            return list(), list(), list(), list()
        df = df.loc[idx]
    x_e, y_e, x_f, y_f = dataframe_to_tuples(df,
                                             **kwargs)
    p_e = model.predict(x_e)
    p_f = model.predict(x_f)
    return y_e, p_e, y_f, p_f


def batched_prediction(model: WeightedLinearModel,
                       filename: str,
                       table_names: Collection = None,
                       subset_keys: Collection = None,
                       drop_columns: List[str] = None,
                       **kwargs):
    """
    Convenience function for optimization workflow. Read inputs/outputs
    from HDF5 file and predict using fitted model.

    Args:
        filename (str): path to HDF5 file.
        model (WeightedLinearModel): fitted model.
        table_names (list): list of table names to query from HDF5 file.
        subset_keys (list): list of keys to query from DataFrame.
        drop_columns (list): list of columns to drop. Used when modifying
            the cutoffs of the feature vectors from HDF5 file. No internal
            checks are performed to see if dropping provided columns produce
            features of the intended cutoffs. Use with Caution.

    Returns:
        y_e (np.ndarray): target values for energies.
        p_e (np.ndarray): prediction values for energies.
        y_f (np.ndarray): target values for forces.
        p_f (np.ndarray): prediction values for forces.
    """
    if table_names is None:
        _, _, table_names, _ = io.analyze_hdf_tables(filename)
    df_batches = io.dataframe_batch_loader(filename, table_names)
    y_e = []
    p_e = []
    y_f = []
    p_f = []
    for df in df_batches:
        if drop_columns != None:
            df.drop(columns=drop_columns,inplace=True)

        predictions = subset_prediction(df,
                                        model,
                                        subset_keys=subset_keys,
                                        **kwargs)
        y_e.append(predictions[0])
        p_e.append(predictions[1])
        y_f.append(predictions[2])
        p_f.append(predictions[3])
    y_e = np.concatenate(y_e)
    p_e = np.concatenate(p_e)
    y_f = np.concatenate(y_f)
    p_f = np.concatenate(p_f)
    return y_e, p_e, y_f, p_f


def rmse_metric(predicted: Collection,
                actual: Collection) -> float:
    """
    Root-mean-square error metric.

    Args:
        predicted (list): prediction values.
        actual (list): reference values.

    Returns:
        root-mean-square-error metric.
    """
    return np.sqrt(np.mean(np.subtract(predicted, actual) ** 2))


def mae_metric(predicted, actual):
    """
    Mean-absolute error metric.

    Args:
        predicted (list): prediction values.
        actual (list): reference values.

    Returns:
        mean-absolute error metric.
    """
    return np.mean(np.abs(np.subtract(predicted, actual)))


def arrange_coefficients(coefficients, bspline_config):
    """
    Arrange coefficients by degree of interaction.

    Args:
        coefficients (np.ndarray): Flattened vector of coefficients.
            Partitioned by provided bspline_config per degree.
        bspline_config (bspline.BSplineBasis)

    Returns:
        solutions (dict): fit coefficients per degree.
    """
    split_indices = np.cumsum(bspline_config.partition_sizes)[:-1]
    solutions_list = np.array_split(coefficients,
                                    split_indices)
    element_list = bspline_config.element_list
    solutions = {element: value[0] for element, value
                 in zip(element_list, solutions_list[:len(element_list)])}
    solutions_list = solutions_list[len(element_list):]

    j = 0
    for d in range(2, bspline_config.degree + 1):
        interactions_map = bspline_config.interactions_map[d]
        for interaction in interactions_map:
            solutions[interaction] = solutions_list[j]
            j += 1
    return solutions


def postprocess_coefficients_2b(coefficients,
                                core_hardness=2.0,
                                min_core=2.0,
                                min_slope=0.1,
                                rounding_factor=3,
                                smooth_cutoff=False,
                                in_place=False):
    """
    Postprocess 2B coefficients to enforce repulsive core.

    Args:
        coefficients (np.ndarray): vector of 2B coefficients.
        core_hardness (float): power base factor for hard-core correction.
        min_core (float): minimum energy barrier at the lower-bound (eV).
        min_slope (float): minimum core slope at peak (eV).
        rounding_factor (float): decimal for rounding in extrema search.
        smooth_cutoff (bool): whether to fix the last two coefficients to
            zero, forcing the second derivative to be zero at the upper bound.
        in_place (bool): whether to modify in-place or make a copy.

    Returns:
        coefficients (np.ndarray): new vector of coefficients.
    """
    if not in_place:  # apply corrections to a copy
        coefficients = np.array(coefficients)
    well_idx = find_pair_potential_well(coefficients, rounding_factor)
    if well_idx > 1:
        # search for maximum left of potential well, rounding to meV (default)
        peak_search = np.round(coefficients[:well_idx], rounding_factor)
        # bias towards well with imperceptible slope to deal with plateau
        peak_search += np.arange(len(peak_search)) * 10**(-2 * rounding_factor)
        gradient = np.gradient(peak_search)
        peak_idx = np.argmax(peak_search)
        if np.all(gradient[:peak_idx] >= 0):
            # correction for case where lower-bound is far below
            # observations and coefficients are nearly zero.
            for i in np.arange(peak_idx)[::-1]:
                value = np.abs(coefficients[i + 1]) * core_hardness
                value = max(value, min_slope)
                coefficients[i] = value
    if coefficients[0] < min_core:
        # fail-safe hard core by simply fixing the first coefficient
        coefficients[0] = min_core
    if smooth_cutoff:
        coefficients[-2:] = 0
    return coefficients


def find_pair_potential_well(coefficients, rounding_factor):
    """
    Identify coefficient index corresponding to possible potential well.
    Intermediate function for postprocess_coefficients_2b().

    Args:
        coefficients: vector of two-body coefficients.
        rounding_factor: decimal for rounding in extrema search.

    Returns:
        well_idx: approximate location of potential well in coefficients
    """
    peak_idx = np.argmax(coefficients)
    well_idx = np.argmin(coefficients)
    if well_idx < peak_idx:
        # if well is left of peak, either core may not be well-defined
        # or well may not be well-defined
        well_search = np.round(coefficients[:peak_idx], rounding_factor)
        if np.ptp(well_search) < 10 ** -(rounding_factor - 1):
            # no actual well
            well_idx = peak_idx + 1
    return well_idx


def calc_E_F_weights(n_e, n_f, std_e, std_f):
    """
    Calculates weights applied to energy and force components of the
    least-squares problem (excluding kappa, which is applied in
    self.combine_weighted_gram()).

    Args:
        n_e (int): number of energy samples.
        n_f (int): number of force samples.
        e_stddev (float): standard deviation of energy samples.
        f_stddev (float): standard deviation of force samples.

    Returns:
        energy_weight (float): weight applied to energy components.
        force_weight (float): weight applied to force components.
    """
    if std_e == 0:  # single point or really bad dataset
        energy_weight = 1.0
        force_weight = 1 / np.sqrt(n_f)
    else:
        energy_weight = 1 / np.sqrt(n_e) / std_e
        force_weight = 1 / np.sqrt(n_f) / std_f
    return energy_weight, force_weight