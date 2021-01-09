import pytest
import ase
from uf3.representation.process import *
from uf3.data import composition
from uf3.data import io


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
    chemistry_config = composition.ChemicalSystem(element_list,
                                                  r_min_map={pair: 2.9},
                                                  r_max_map={pair: 5.1},
                                                  resolution_map={pair: 15})
    yield chemistry_config


@pytest.fixture()
def water_chemistry():
    element_list = ['H', 'O']
    chemistry_config = composition.ChemicalSystem(element_list)
    yield chemistry_config


@pytest.fixture()
def binary_chemistry():
    element_list = ['Ne', 'Xe']
    chemistry_config = composition.ChemicalSystem(element_list)
    yield chemistry_config


class TestBasis:
    def test_setup(self, unary_chemistry):
        bspline_handler = BasisProcessor2B(unary_chemistry)
        assert bspline_handler.r_cut == 5.1
        assert len(bspline_handler.knots_map[('Ar', 'Ar')]) == 22
        assert len(bspline_handler.knot_subintervals[('Ar', 'Ar')]) == 18
        assert bspline_handler.n_features == 18  # 15 + 3
        assert len(bspline_handler.columns) == 20  # 1 + 18 + 1

    def test_regularizer_subdivision(self, binary_chemistry):
        bspline_handler = BasisProcessor2B(binary_chemistry)
        subdivisions = bspline_handler.get_regularizer_sizes()
        # default 20 intervals yields 23 basis functions
        assert np.allclose(subdivisions, [23, 23, 23, 2])

    def test_energy_features(self, unary_chemistry, simple_molecule):
        bspline_handler = BasisProcessor2B(unary_chemistry)
        vector = bspline_handler.get_energy_features(simple_molecule,
                                                     simple_molecule)
        assert len(vector) == 18 + 1  # 18 features and one 1-body term

    def test_force_features(self, unary_chemistry, simple_molecule):
        bspline_handler = BasisProcessor2B(unary_chemistry)
        vector = bspline_handler.get_force_features(simple_molecule,
                                                    simple_molecule)
        assert vector.shape == (3, 3, 19)  # 3 forces per atom

    def test_evaluate_single(self, unary_chemistry, simple_molecule):
        bspline_handler = BasisProcessor2B(unary_chemistry)
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
        bspline_handler = BasisProcessor2B(unary_chemistry)
        df = pd.DataFrame(columns=['geometry', 'energy', 'fx', 'fy', 'fz'])
        df.loc[0] = [None,
                     1.5,
                     [4, 3, 0],
                     [0, 1, 2],
                     [2, 1, 0]]
        df.at[0, 'geometry'] = simple_molecule
        data_coordinator = io.DataCoordinator()
        df_features = bspline_handler.evaluate(df, data_coordinator)
        assert len(df_features) == 1 + 3 * 3  # energy and 3 forces per atom
        assert len(df_features.columns) == 1 + 18 + 1
        x, y, w = bspline_handler.get_training_tuples(df_features,
                                                      0.5,
                                                      data_coordinator)
        assert x.shape == (1 + 3 * 3, 18 + 1)
        assert np.allclose(y, [1.5, 4, 3, 0, 0, 1, 2, 2, 1, 0])

    def test_evaluate_binary(self, water_chemistry, simple_water):
        bspline_handler = BasisProcessor2B(water_chemistry)
        df = pd.DataFrame(columns=['geometry', 'energy', 'fx', 'fy', 'fz'])
        df.loc[0] = [None,
                     1.5,
                     [4, 3, 0],
                     [0, 1, 2],
                     [2, 1, 0]]
        df.at[0, 'geometry'] = simple_water
        data_coordinator = io.DataCoordinator()
        df_feats = bspline_handler.evaluate(df, data_coordinator)
        assert len(df_feats) == 1 + 3 * 3  # energy and 3 forces per atom
        assert len(df_feats.columns) == 1 + 23 * 3 + 2
        # energy, 23 features per interaction, two 1-body terms
        x, y, w = bspline_handler.get_training_tuples(df_feats,
                                                      0.5,
                                                      data_coordinator)
        assert x.shape == (1 + 3 * 3, 23 * 3 + 2)
        assert np.allclose(y, [1.5, 4, 3, 0, 0, 1, 2, 2, 1, 0])


def test_flatten_by_interactions():
    vector_map = {('A', 'A'): np.array([1, 1, 1]),
                  ('A', 'B'): np.array([2, 2]),
                  ('B', 'B'): np.array([3, 3, 3, 3])}
    pair_tuples = [('A', 'A'), ('A', 'B'), ('B', 'B')]
    vector = flatten_by_interactions(vector_map, pair_tuples)
    assert np.allclose(vector, [1, 1, 1, 2, 2, 3, 3, 3, 3])