import pytest

import uf3.representation.bspline
from uf3.representation.bspline import *
from uf3.data import composition

@pytest.fixture()
def binary_chemistry():
    element_list = ['Ne', 'Xe']
    chemistry_config = composition.ChemicalSystem(element_list)
    yield chemistry_config


class TestKnots:
    def test_knot_sequence_from_points(self):
        sequence = knot_sequence_from_points([1, 2, 3])
        assert np.allclose(sequence, [1, 1, 1, 1, 2, 3, 3, 3, 3])

    def test_get_knot_subintervals(self):
        sequence = knot_sequence_from_points([1, 2, 3])
        subintervals = get_knot_subintervals(sequence)
        assert np.allclose(subintervals[0], [1, 1, 1, 1, 2])
        assert np.allclose(subintervals[2], [1, 1, 2, 3, 3])
        assert np.allclose(subintervals[4], [2, 3, 3, 3, 3])

    def test_generate_uniform_knots(self):
        points = generate_uniform_knots(1, 6, 5, sequence=False)
        sequence = generate_uniform_knots(1, 6, 5, sequence=True)
        assert np.allclose(points, [1, 2, 3, 4, 5, 6])
        assert np.allclose(sequence, [1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6])

    def test_lammps_knots(self):
        points = generate_lammps_knots(0, 1, 2)
        points = np.round(points, 4)
        assert np.allclose(points, [0, 0, 0, 0, 0.7071, 1, 1, 1, 1])


class TestBSplineConfig:
    def test_regularizer_subdivision(self, binary_chemistry):
        bspline_handler = BSplineBasis(binary_chemistry)
        partitions = bspline_handler.get_feature_partition_sizes()
        # default 15 intervals yields 18 basis functions
        assert np.allclose(partitions, [1, 1, 18, 18, 18])

    def test_custom_knots(self):
        element_list = ['Au', 'Ag']
        chemistry = composition.ChemicalSystem(element_list)
        knots_map = {('Ag', 'Au'): [1, 1, 1, 1, 1.1, 1.1, 1.1, 1.1]}
        bspline_handler = BSplineBasis(chemistry,
                                       knots_map=knots_map)
        assert bspline_handler.r_min_map[('Ag', 'Au')] == 1.0
        assert bspline_handler.r_max_map[('Ag', 'Au')] == 1.1
        assert bspline_handler.resolution_map[('Ag', 'Au')] == 1
        assert bspline_handler.r_min_map[('Au', 'Au')] == 1.0
        assert bspline_handler.r_max_map[('Au', 'Au')] == 8.0
        assert bspline_handler.resolution_map[('Au', 'Au')] == 15

    def test_unary(self):
        element_list = ['Au']
        chemistry = composition.ChemicalSystem(element_list)
        bspline_handler = BSplineBasis(chemistry,
                                       r_min_map={('Au', 'Au'): 1.1})
        assert bspline_handler.r_min_map[('Au', 'Au')] == 1.1
        assert bspline_handler.r_max_map[('Au', 'Au')] == 8.0
        assert bspline_handler.resolution_map[('Au', 'Au')] == 15

    def test_binary(self):
        element_list = ['Ne', 'Xe']
        chemistry = composition.ChemicalSystem(element_list)
        bspline_handler = BSplineBasis(chemistry,
                                       resolution_map={('Ne', 'Xe'): 10})
        assert bspline_handler.r_min_map[('Ne', 'Ne')] == 1.0
        assert bspline_handler.r_max_map[('Xe', 'Xe')] == 8.0
        assert bspline_handler.resolution_map[('Ne', 'Xe')] == 10

    def test_regularizer(self):
        ridge_map = {1: 2, 2: 0.5}
        curvature_map = {2: 1}
        element_list = ['Ne', 'Xe']
        chemistry = composition.ChemicalSystem(element_list)
        bspline_handler = BSplineBasis(chemistry)
        matrix = bspline_handler.get_regularization_matrix(ridge_map,
                                                           curvature_map)
        ridge_sum = (2 * 2) + (0.5 * (18 + 18 + 18))
        curv_sum = (0 * 2) + (1 + (2 * (18-2)) + 1) * 3
        assert np.sum(matrix) == ridge_sum
        assert np.sum(np.diag(matrix)) == ridge_sum + curv_sum

        matrix = bspline_handler.get_regularization_matrix(r1=2,
                                                           r2=0.5,
                                                           c2=1)
        ridge_sum = (2 * 2) + (0.5 * (18 + 18 + 18))
        curv_sum = (0 * 2) + (1 + (2 * (18-2)) + 1) * 3
        assert np.sum(matrix) == ridge_sum
        assert np.sum(np.diag(matrix)) == ridge_sum + curv_sum

def test_fit_spline_1d():
    x = np.linspace(-1, 7, 1000)
    y = np.sin(x) + 0.5 * x
    knot_sequence = uf3.representation.bspline.generate_lammps_knots(0, 6, 5)
    coefficients = fit_spline_1d(x, y, knot_sequence)
    coefficients = np.round(coefficients, 2)
    assert np.allclose(coefficients,
                       [-0.06, 1.59, 2.37, 1.16, 1.23, 1.77, 2.43, 2.71])
    bspline = interpolate.BSpline(t=knot_sequence,
                                  c=coefficients,
                                  k=3,
                                  extrapolate=False)
    yp = bspline(x[(x > 0) & (x < 6)])
    rmse = np.sqrt(np.mean(np.subtract(y[(x > 0) & (x < 6)], yp) ** 2))
    assert rmse < 0.017


def test_distance_bspline():
    points = np.array([1e-10,  # close to 0
                       0.5,
                       1 - 1e-10])  # close to 1
    sequence = uf3.representation.bspline.knot_sequence_from_points([0, 1])
    subintervals = uf3.representation.bspline.get_knot_subintervals(sequence)
    basis_functions = generate_basis_functions(subintervals)
    values_per_spline = evaluate_basis_functions(points,
                                                 basis_functions,
                                                 flatten=False)
    assert len(values_per_spline) == 4
    assert len(values_per_spline[0]) == 3
    assert np.allclose(values_per_spline[0], [1, 0.125, 0])
    assert np.allclose(values_per_spline[1], [0, 0.375, 0])
    assert np.allclose(values_per_spline[2], [0, 0.375, 0])
    assert np.allclose(values_per_spline[3], [0, 0.125, 1])
    value_per_spline = evaluate_basis_functions(points, basis_functions)
    assert len(value_per_spline) == 4
    assert np.allclose(value_per_spline, [1.125, 0.375, 0.375, 1.125])


def test_force_bspline():
    distances = np.array([3, 4, 3, 5, 4, 5])  # three atom molecular triangle
    drij_dR = np.array([[[-1.0, -0.0, -1.0, -0.0, 0.0, 0.0, ],
                         [-0.0, -1.0, 0.0, 0.0, -1.0, -0.0, ],
                         [-0.0, -0.0, 0.0, 0.0, 0.0, 0.0, ]],
                        [[1.0, 0.0, 1.0, 0.6, 0.0, 0.6],
                         [0.0, 0.0, -0.0, -0.8, -0.0, -0.8],
                         [0.0, 0.0, -0.0, -0.0, 0.0, 0.0, ]],
                        [[0.0, 0.0, -0.0, -0.6, -0.0, -0.6],
                         [0.0, 1.0, 0.0, 0.8, 1.0, 0.8],
                         [0.0, 0.0, 0.0, 0.0, -0.0, -0.0, ]]])

    sequence = uf3.representation.bspline.knot_sequence_from_points([2, 6])
    subintervals = uf3.representation.bspline.get_knot_subintervals(sequence)
    basis_functions = generate_basis_functions(subintervals)
    x = featurize_force_2B(basis_functions, distances, drij_dR, sequence)
    assert x.shape == (3, 3, 4)
    assert np.ptp(x[:, 2, :]) == 0  # no z-direction component
    assert np.ptp(np.sum(x, axis=0)) < 1e-10  # forces cancel along atom axis
    assert np.any(np.ptp(x, axis=0) > 0)  # but should not be entirely zero
    assert np.ptp(np.sum(x, axis=2)) < 1e-10  # values cancel across b-splines
    assert np.any(np.ptp(x, axis=2) > 0)  # but should not be entirely zero
