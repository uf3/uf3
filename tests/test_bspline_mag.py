import pytest

import uf3.representation.bspline
from uf3.representation.bspline import *
from uf3.data import composition


def test_distance_bspline_mag():
    points = np.array([1e-10,  0.5,   1 - 1e-10])  # close to 1
    mag_points = [list([1, 2]), list([-1, 2]), list([2, -1])]
    sequence = uf3.representation.bspline.knot_sequence_from_points([0, 1])
    subintervals = uf3.representation.bspline.get_knot_subintervals(sequence)
    basis_functions = generate_basis_functions(subintervals)
    mag_terms = ['exchange', 'self_quadratic', 'self_biquadratic']
    value_per_spline_M = {}
    for mag_type in mag_terms:
        value_per_spline_M[mag_type] = evaluate_basis_functions_w_magnetism(points,
                                                                            mag_points,
                                                                            mag_type,
                                                                            basis_functions,
                                                                            flatten=False)
    for mag_type in mag_terms:
        assert len(value_per_spline_M[mag_type]) == 4
        assert len(value_per_spline_M[mag_type][0]) == 3
    assert np.allclose(value_per_spline_M['exchange'][0], [2, -0.25, 0])
    
    # Not done yet


def test_force_bspline_mag():
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
    magmom_list = [-3, 2, -1] # magmetic moments of the pseudo atoms(-3,2,-1), corresponding to the distance map
    mag_points = [list([-3, 2]), list([-3, -1]), list([2, -3]), list([2, -1]), list([-1, -3]), list([-1, 2])]
    mag_terms = ['exchange', 'self_quadratic', 'self_biquadratic']
    sequence = uf3.representation.bspline.knot_sequence_from_points([2, 4, 6])
    subintervals = uf3.representation.bspline.get_knot_subintervals(sequence)
    basis_functions = generate_basis_functions(subintervals)
    f_ma = {}
    f_ra = {}
    for mag_type in mag_terms:
        f_ma[mag_type] = evaluate_force_2B_w_magnetism_ma(distances,
                                                          mag_points,
                                                          mag_type,
                                                          basis_functions,
                                                          flatten=True)
        f_ra[mag_type] = featurize_force_2B_w_magnetism_ra(basis_functions,
                                                           distances,
                                                           drij_dR,
                                                           mag_type,
                                                           mag_points,
                                                           magmom_list,
                                                           sequence)
    for mag_type in mag_terms:
        assert len(f_ma[mag_type]) == 5
        assert f_ra[mag_type].shape == (3, 3, 5)
        assert np.ptp(f_ra[mag_type][:, 2, :]) == 0  # no z-direction component
        assert np.ptp(np.sum(f_ra[mag_type], axis=0)) < 1e-10  # forces cancel along atom axis
        assert np.any(np.ptp(f_ra[mag_type], axis=0) > 0)  # but should not be entirely zero
        assert np.ptp(np.sum(f_ra[mag_type], axis=2)) < 1e-10  # values cancel across b-splines
        assert np.any(np.ptp(f_ra[mag_type], axis=2) > 0)  # but should not be entirely zero
    # Not done yet, assert np.ptp(np.sum(f_ra[mag_type], axis=0)) < 1e-10 always gives False