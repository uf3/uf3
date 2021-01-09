import numpy as np
from uf3.regression import least_squares
from uf3.regression import regularize


def simple_problem(n_features, n_samples, seed=0):
    np.random.seed(seed)
    x = np.random.rand(n_samples, n_features)
    c = np.random.rand(n_features)
    y = np.dot(x, c)
    return x, y, c


class TestLinearModel:
    def test_init(self):
        regularizer = regularize.Regularizer(regularizer_sizes=[9, 9, 2],
                                             ridge=0.5,
                                             curvature=1,
                                             onebody=2)
        model = least_squares.WeightedLinearModel(regularizer=regularizer)
        assert model.regularizer.shape == (20, 20)
        model = least_squares.WeightedLinearModel(regularizer_sizes=[9, 9, 2])
        assert model.regularizer.shape == (20, 20)

    def test_fit_predict_score(self):
        x, y, c = simple_problem(20, 500, seed=0)
        model = least_squares.WeightedLinearModel(regularizer_sizes=[9, 9, 2],
                                                  curvature=0,
                                                  ridge=1e-4)
        model.fit(x, y)
        assert np.allclose(model.coefficients, c)
        assert np.allclose(model.predict(x), y)
        assert model.score(x, y) < 1e-6


def test_linear_least_squares():
    x, y, c = simple_problem(10, 30, seed=0)
    solution = least_squares.linear_least_squares(x, y)
    assert np.allclose(solution, c)


def test_weighted_least_squares():
    x1, y1, c1 = simple_problem(30, 100, seed=0)
    x2, y2, c2 = simple_problem(30, 200, seed=1)
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])

    weights = np.concatenate([np.ones(100), np.zeros(200)])
    solution = least_squares.weighted_least_squares(x, y, weights)
    assert np.allclose(solution, c1)
    weights = np.concatenate([np.zeros(100), np.ones(200)])
    solution = least_squares.weighted_least_squares(x, y, weights)
    assert np.allclose(solution, c2)
    weights = np.concatenate([np.ones(100) * 0.5, np.ones(200) * 0.5])
    solution = least_squares.weighted_least_squares(x, y, weights)
    assert not np.allclose(solution, c1) and not np.allclose(solution, c2)
    weights = np.concatenate([np.ones(100) * 0.5, np.ones(200) * 0.5])
    solution = least_squares.weighted_least_squares(x, y, weights,
                                                    fixed=[(0, 10),
                                                           (3, 4),
                                                           (5, 0)])
    assert solution[0] == 10
    assert solution[3] == 4
    assert solution[5] == 0


def test_fixed_coefficients():
    x = np.random.rand(100, 30)
    y = np.random.rand(100)
    regularizers = [np.random.rand(30, 30)]
    coefficients = [3, 4, 5]
    colidx = [10, 15, 20]
    xf, yf, mask = least_squares.preprocess_fixed_coefficients(x,
                                                               y,
                                                               regularizers,
                                                               coefficients,
                                                               colidx)
    assert xf.shape == (100, 27)
    assert np.sum(yf) != np.sum(y)
    assert len(mask) == 27
    assert np.all([(idx not in mask) for idx in colidx])
