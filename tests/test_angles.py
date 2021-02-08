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
    knot_subintervals = knots.get_knot_subintervals(knot_sequence)
    basis_functions = bspline.generate_basis_functions(knot_subintervals)
    grid_3b = compute_force_3b(simple_molecule,
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