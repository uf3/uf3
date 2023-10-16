import numpy as np
import uf3
from uf3.regression import least_squares
from uf3.data import composition
from uf3.representation import bspline

import os


def simple_problem(n_features, n_samples, seed=0):
    np.random.seed(seed)
    x = np.random.rand(n_samples, n_features)
    c = np.random.rand(n_features)
    y = np.dot(x, c)
    return x, y, c


class TestLinearModel:
    def test_init(self):
        regularizer = np.eye(20)
        model = least_squares.BasicLinearModel(regularizer=regularizer)
        assert model.regularizer.shape == (20, 20)

    def test_fit_predict_score(self):
        x, y, c = simple_problem(20, 500, seed=0)
        regularizer = np.eye(20) * 1e-6
        model = least_squares.BasicLinearModel(regularizer=regularizer)
        model.fit(x, y)
        assert np.allclose(model.coefficients, c)
        assert np.allclose(model.predict(x), y)
        assert model.score(x, y) < 1e-6



def test_linear_least_squares():
    x, y, c = simple_problem(10, 30, seed=0)
    solution = least_squares.linear_least_squares(x, y)
    assert np.allclose(solution, c)


def test_weighted_least_squares():
    x1, y1, c1 = simple_problem(5, 10, seed=0)
    x2, y2, c2 = simple_problem(5, 20, seed=1)
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])

    weights = np.concatenate([np.ones(10), np.zeros(20)])
    solution = least_squares.weighted_least_squares(x, y, weights)
    assert np.allclose(solution, c1)
    weights = np.concatenate([np.zeros(10), np.ones(20)])
    solution = least_squares.weighted_least_squares(x, y, weights)
    assert np.allclose(solution, c2)
    weights = np.concatenate([np.ones(10) * 0.5, np.ones(20) * 0.5])
    solution = least_squares.weighted_least_squares(x, y, weights)
    assert not np.allclose(solution, c1) and not np.allclose(solution, c2)


def test_frozen_coefficients():
    n_dims = 5
    x1, y1, c1 = simple_problem(n_dims, 10, seed=0)
    x2, y2, c2 = simple_problem(n_dims, 20, seed=1)
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    r = np.eye(n_dims) * 1e-6

    weights = np.concatenate([np.ones(10) * 0.5, np.ones(20) * 0.5])

    fixed_tuples = np.array([(0, 10), (2, 4), (4, 0)])
    col_idx = fixed_tuples[:, 0]
    frozen_c = fixed_tuples[:, 1]

    mask = least_squares.get_freezing_mask(n_dims, col_idx)
    r = least_squares.freeze_regularizer(r, mask)
    x, y = least_squares.freeze_columns(x, y, mask, frozen_c, col_idx)

    solution = least_squares.weighted_least_squares(x, y, weights,
                                                    regularizer=r)
    solution = least_squares.revert_frozen_coefficients(solution,
                                                        n_dims,
                                                        mask,
                                                        frozen_c,
                                                        col_idx)
    assert solution[0] == 10
    assert solution[2] == 4
    assert solution[4] == 0


# def test_interpret_regularizer():
#     args = ["c_2b", "r_1b", "r_2b"]  # d = 2
#     r = least_squares.interpret_regularizer([0], 2)
#     assert np.allclose([r[k] for k in args], [1, 1e-5, 0])
#     args = ["c_2b", "c_3b", "r_1b", "r_2b", "r_3b"]  # d = 3
#     r = least_squares.interpret_regularizer([3, -1], 3)
#     assert np.allclose([r[k] for k in args], [1000, 0.1, 1e-5, 0, 0])

def test_singlepoint_fit():
    chemical_system = composition.ChemicalSystem(["Al"])
    bspline_config = bspline.BSplineBasis(chemical_system)
    n_features = sum(bspline_config.partition_sizes)
    x_e, y_e, _ = simple_problem(n_features, 1, seed=0)  # single point
    x_f, y_f, _ = simple_problem(n_features, 3, seed=1)
    regularizer = np.eye(n_features) * 1e-6
    model = least_squares.WeightedLinearModel(bspline_config,
                                              regularizer=regularizer)
    model.fit(x_e, y_e, x_f, y_f)
    assert sum(~np.isfinite(model.coefficients)) == 0  # no nan or inf

def test_singlepoint_fit_from_file():
    chemical_system = composition.ChemicalSystem(["Al"])
    bspline_config = bspline.BSplineBasis(chemical_system)
    pkg_directory = os.path.dirname(os.path.dirname(uf3.__file__))
    data_directory = os.path.join(pkg_directory, "tests/data")
    features = os.path.join(data_directory, "singlepoint_fit",
                            "df_features_test_singlepoint_fit_from_file.h5")
    n_features = 19
    regularizer = np.eye(n_features) * 1e-6
    model = least_squares.WeightedLinearModel(bspline_config,
                                              regularizer=regularizer)
    model.fit_from_file(features,
                        subset=['0_0'])
    assert sum(~np.isfinite(model.coefficients)) == 0  # no nan or inf

def test_loss_function():
    chemical_system = composition.ChemicalSystem(["Al"], degree=2)
    # Don't trim columns for this test
    bspline_config = bspline.BSplineBasis(chemical_system,
                                          leading_trim=0,
                                          trailing_trim=0)
    n_features = sum(bspline_config.partition_sizes)
    n_samples_e = 30
    n_samples_f = 500
    x, y_true, _ = simple_problem(n_features, n_samples_e + n_samples_f, seed=0)
    noise = np.random.normal(0, 0.1, n_samples_e + n_samples_f)
    y = y_true + noise
    x_e = x[:n_samples_e]
    y_e = y[:n_samples_e]
    x_f = x[n_samples_e:]
    y_f = y[n_samples_e:]
    regularizer = np.zeros((n_features, n_features))  # no regularizer for this test
    model = least_squares.WeightedLinearModel(bspline_config,
                                              regularizer=regularizer)
    kappa = 0.25
    model.fit(x_e, y_e, x_f, y_f, weight=kappa)

    # Test if the fitted coefficients are the minimum to the loss function
    e_weight = kappa / len(y_e) / np.var(y_e)
    f_weight = (1 - kappa) / len(y_f) / np.var(y_f)
    def loss_function(c):
        e_loss = np.sum((y_e - np.dot(x_e, c)) ** 2) * e_weight
        f_loss = np.sum((y_f - np.dot(x_f, c)) ** 2) * f_weight
        return e_loss + f_loss

    # loss evaluation for given coefficients
    c_ref = model.coefficients
    loss_ref = loss_function(c_ref)

    # loss evaluation for perturbed coefficients
    for i in range(len(c_ref)):
        c = c_ref.copy()
        c[i] += 1e-6
        loss = loss_function(c)
        assert loss > loss_ref
        c[i] -= 2e-6
        loss = loss_function(c)
        assert loss > loss_ref

    # loss evaluation for random perturbations
    for i in range(10):
        c = c_ref + np.random.normal(0, 1e-6, len(c_ref))
        loss = loss_function(c)
        assert loss > loss_ref
