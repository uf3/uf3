import pytest
import ase
from uf3.representation.process import *
from uf3.representation import knots
from uf3.representation import bspline
from uf3.data import composition
from uf3.data import io

from uf3.representation.angles import *


@pytest.fixture()
def simple_molecule():
    geom = ase.Atoms('Ar3',
                     positions=[[0, 0, 0], [3, 0.0, 0.0], [0, 4.0, 0]],
                     pbc=False,
                     cell=None)
    yield geom


def test_compute_force(simple_molecule):
    knot_sequence = [2.5, 2.5, 2.5,
                     2.5, 5.5,
                     5.5, 5.5, 5.5]
    knot_sequence = np.array(knot_sequence)
    knot_subintervals = knots.get_knot_subintervals(knot_sequence)
    basis_functions = bspline.generate_basis_functions(knot_subintervals)
    grid_3b = featurize_force_3B(simple_molecule,
                                 simple_molecule,
                                 knot_sequence,
                                 basis_functions)
    assert len(grid_3b) == 3
    assert len(grid_3b[0]) == 3
    x = np.array([[c_grid.flatten() for c_grid in a_grid]
                  for a_grid in grid_3b])
    assert np.sum(x) != 0
    assert np.ptp(x[:, 2, :]) == 0  # no z-direction component
    assert np.ptp(np.sum(x, axis=0)) < 1e-10  # forces cancel along atom axis
    assert np.any(np.ptp(x, axis=0) > 0)  # but should not be entirely zero
    assert np.ptp(np.sum(x, axis=2)) < 1e-10  # values cancel across b-splines
    assert np.any(np.ptp(x, axis=2) > 0)  # but should not be entirely zero


def test_spline_3b():
    # import os
    # os.environ["NUMBA_DUMP_OPTIMIZED"] = "1"
    # import llvmlite.binding as llvm
    # llvm.set_option('', '--debug-only=loop-vectorize')
    triangle_values = np.array([[1.1, 2.0, 3.2],
                                [1.2, 2.1, 3.3],
                                [1.3, 2.2, 3.4],
                                [1.4, 2.3, 3.5],
                                [4.1, 5.0, 6.2],
                                [4.2, 5.1, 6.3],
                                [4.3, 5.2, 6.4],
                                [4.4, 5.3, 6.5]
                                ])
    idx_rl = np.array([0, 1, 2, 3, 1, 2, 3, 4])
    idx_rm = np.array([0, 1, 2, 3, 1, 2, 3, 4])
    idx_rn = np.array([0, 1, 2, 3, 1, 2, 3, 4])
    knot_sequence = np.array([0.0, 0, 0, 1, 2, 3, 3, 3, 3.0])
    L = len(knot_sequence) - 4

    old = arrange_3B(triangle_values, idx_rl, idx_rm, idx_rn, L)
    # spline_3b.parallel_diagnostics(level=4)
    # spline_3b.inspect_types()
    # print()
    # L = len(knot_sequence)-4
    # triangle_groups = partition_triangles(triangle_values,
    #                                       idx_rl,
    #                                       idx_rm,
    #                                       idx_rn,
    #                                       L)
    # subgrids = [evaluate_local_triangles(group) for group in triangle_groups]
    # new = arrange_subgrids(subgrids, L)
    # for idx, group in enumerate(triangle_groups):
    #     v = np.unravel_index(idx, (L - 3, L - 3, L - 3))
    #     print(v, group)
    # for idx, group in enumerate(subgrids):
    #     v = np.unravel_index(idx, (L - 3, L - 3, L - 3))
    #     print(v, group)
    # assert np.allclose(old, new)



