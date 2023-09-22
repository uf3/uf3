import numpy as np
import uf3
from uf3.regression import least_squares
from uf3.regression import regularize
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
    from uf3.data import composition
    from uf3.representation import bspline
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
    from uf3.data import composition
    from uf3.representation import bspline
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