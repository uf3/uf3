import pytest
import ase
from ufpotential.bspline.basis import *
from ufpotential.bspline import knots
from ufpotential.data import chemistry
from ufpotential.data import io


@pytest.fixture()
def simple_molecule():
    geom = ase.Atoms('Ar3',
                     positions=[[0, 0, 0], [3, 0.0, 0.0], [0, 4.0, 0]],
                     pbc=False,
                     cell=None)
    yield geom


@pytest.fixture()
def simple_water():
    geom = ase.Atoms('H2O',
                     positions=[[0, 0, 0], [3, 0.0, 0.0], [0, 4.0, 0]],
                     pbc=False,
                     cell=None)
    yield geom



@pytest.fixture()
def unary_chemistry():
    element_list = ['Ar']
    pair = ('Ar', 'Ar')
    chemistry_config = chemistry.ChemistryConfig(element_list,
                                                 r_min_map={pair: 2.9},
                                                 r_max_map={pair: 5.1},
                                                 resolution_map={pair: 15})
    yield chemistry_config


@pytest.fixture()
def water_chemistry():
    element_list = ['H', 'O']
    chemistry_config = chemistry.ChemistryConfig(element_list)
    yield chemistry_config


@pytest.fixture()
def binary_chemistry():
    element_list = ['Ne', 'Xe']
    chemistry_config = chemistry.ChemistryConfig(element_list)
    yield chemistry_config


class TestBasis:
    def test_setup(self, unary_chemistry):
        bspline_handler = Bspline1DBasis(unary_chemistry)
        assert bspline_handler.r_cut == 5.1
        assert len(bspline_handler.knots_map[('Ar', 'Ar')]) == 22
        assert len(bspline_handler.knot_subintervals[('Ar', 'Ar')]) == 18
        assert bspline_handler.n_features == 18  # 15 + 3
        assert len(bspline_handler.columns) == 20  # 1 + 18 + 1

    def test_regularizer_subdivision(self, binary_chemistry):
        bspline_handler = Bspline1DBasis(binary_chemistry)
        subdivisions = bspline_handler.get_regularizer_sizes()
        # default 20 intervals yields 23 basis functions
        assert np.allclose(subdivisions, [23, 23, 23, 2])

    def test_energy_features(self, unary_chemistry, simple_molecule):
        bspline_handler = Bspline1DBasis(unary_chemistry)
        vector = bspline_handler.get_energy_features(simple_molecule,
                                                     simple_molecule)
        assert len(vector) == 18 + 1  # 18 features and one 1-body term

    def test_force_features(self, unary_chemistry, simple_molecule):
        bspline_handler = Bspline1DBasis(unary_chemistry)
        vector = bspline_handler.get_force_features(simple_molecule,
                                                    simple_molecule)
        assert vector.shape == (3, 3, 19)  # 3 forces per atom

    def test_evaluate_single(self, unary_chemistry, simple_molecule):
        bspline_handler = Bspline1DBasis(unary_chemistry)
        eval_map = bspline_handler.evaluate_configuration(simple_molecule,
                                                          energy=1.5)
        assert len(eval_map['energy']) == 1 + 18 + 1  # number of columns
        assert eval_map['energy'][0] == 1.5  # energy value
        assert eval_map['energy'][-1] == 3  # scalar for 1-body energy offset
        eval_map = bspline_handler.evaluate_configuration(simple_molecule,
                                                          name='sample',
                                                          forces=[[2, 0, 0],
                                                                  [0, 0, 0],
                                                                  [0, 0, 0]])
        assert len(eval_map) == 3 + 3 + 3  # 3 atoms, 3 forces each
        assert eval_map[('sample', 'fx_0')][0] == 2  # force value
        assert eval_map[('sample', 'fy_1')][-1] == 0  # no 1-body energy offset
        assert len(eval_map[('sample', 'fz_2')]) == 1 + 18 + 1   # columns

    def test_evaluate_unary(self, unary_chemistry, simple_molecule):
        bspline_handler = Bspline1DBasis(unary_chemistry)
        df = pd.DataFrame(columns=['geometry', 'energy', 'fx', 'fy', 'fz'])
        df.loc[0] = [None,
                     1.5,
                     [4, 3, 0],
                     [0, 1, 2],
                     [2, 1, 0]]
        df.at[0, 'geometry'] = simple_molecule
        data_coordinator = io.DataCoordinator()
        df_feats = bspline_handler.evaluate(df, data_coordinator, xy_out=False)
        assert len(df_feats) == 1 + 3 * 3  # energy and 3 forces per atom
        assert len(df_feats.columns) == 1 + 18 + 1
        # energy, 18 features, one 1-body terms
        x, y, u, v = bspline_handler.evaluate(df, data_coordinator)
        assert x.shape == (1, 18 + 1)
        assert u.shape == (3 * 3, 18 + 1)
        assert np.allclose(v, [4, 3, 0, 0, 1, 2, 2, 1, 0])

    def test_evaluate_binary(self, water_chemistry, simple_water):
        bspline_handler = Bspline1DBasis(water_chemistry)
        df = pd.DataFrame(columns=['geometry', 'energy', 'fx', 'fy', 'fz'])
        df.loc[0] = [None,
                     1.5,
                     [4, 3, 0],
                     [0, 1, 2],
                     [2, 1, 0]]
        df.at[0, 'geometry'] = simple_water
        data_coordinator = io.DataCoordinator()
        df_feats = bspline_handler.evaluate(df, data_coordinator, xy_out=False)
        assert len(df_feats) == 1 + 3 * 3  # energy and 3 forces per atom
        assert len(df_feats.columns) == 1 + 23 * 3 + 2
        # energy, 23 features per interaction, two 1-body terms
        x, y, u, v = bspline_handler.evaluate(df, data_coordinator)
        assert x.shape == (1, 23 * 3 + 2)
        assert u.shape == (3 * 3, 23 * 3 + 2)
        assert np.allclose(v, [4, 3, 0, 0, 1, 2, 2, 1, 0])


def test_fit_spline_1d():
    x = np.linspace(-1, 7, 1000)
    y = np.sin(x) + 0.5 * x
    knot_sequence = knots.generate_lammps_knots(0, 6, 5)
    coefficients = fit_spline_1d(x, y, knot_sequence)
    coefficients = np.round(coefficients, 2)
    assert np.allclose(coefficients,
                       [-0.06, 1.59, 2.37, 1.16, 1.23, 1.77, 2.43, 2.71])
    bspline = BSpline(t=knot_sequence, c=coefficients, k=3, extrapolate=False)
    yp = bspline(x[(x > 0) & (x < 6)])
    rmse = np.sqrt(np.mean(np.subtract(y[(x > 0) & (x < 6)], yp) ** 2))
    assert rmse < 0.017


def test_flatten_by_interactions():
    vector_map = {('A', 'A'): np.array([1, 1, 1]),
                  ('A', 'B'): np.array([2, 2]),
                  ('B', 'B'): np.array([3, 3, 3, 3])}
    pair_tuples = [('A', 'A'), ('A', 'B'), ('B', 'B')]
    vector = flatten_by_interactions(vector_map, pair_tuples)
    assert np.allclose(vector, [1, 1, 1, 2, 2, 3, 3, 3, 3])


def test_distance_bspline():
    points = np.array([1e-10,  # close to 0
                       0.5,
                       1 - 1e-10])  # close to 1
    sequence = knots.knot_sequence_from_points([0, 1])
    subintervals = knots.get_knot_subintervals(sequence)
    values_per_spline = evaluate_bspline(points,
                                         subintervals,
                                         flatten=False)
    assert len(values_per_spline) == 4
    assert len(values_per_spline[0]) == 3
    assert np.allclose(values_per_spline[0], [1, 0.125, 0])
    assert np.allclose(values_per_spline[1], [0, 0.375, 0])
    assert np.allclose(values_per_spline[2], [0, 0.375, 0])
    assert np.allclose(values_per_spline[3], [0, 0.125, 1])
    value_per_spline = evaluate_bspline(points, subintervals)
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

    sequence = knots.knot_sequence_from_points([2, 6])
    subintervals = knots.get_knot_subintervals(sequence)

    x = compute_force_bsplines(drij_dR, distances, subintervals)
    assert x.shape == (3, 3, 4)
    assert np.ptp(x[:, 2, :]) == 0  # no z-direction component
    assert np.ptp(np.sum(x, axis=0)) < 1e-10  # forces cancel along atom axis
    assert np.any(np.ptp(x, axis=0) > 0)  # but should not be entirely zero
    assert np.ptp(np.sum(x, axis=2)) < 1e-10  # values cancel across b-splines
    assert np.any(np.ptp(x, axis=2) > 0)  # but should not be entirely zero
