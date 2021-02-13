import pytest
import ase
from uf3.representation.process import *
from uf3.representation import bspline
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
    chemistry_config = composition.ChemicalSystem(element_list)
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
        bspline_config = bspline.BSplineConfig(unary_chemistry)
        bspline_handler = BasisProcessor(unary_chemistry, bspline_config)
        assert bspline_handler.r_cut == 6.0
        assert len(bspline_handler.knots_map[('Ar', 'Ar')]) == 27
        assert len(bspline_handler.basis_functions[('Ar', 'Ar')]) == 23
        assert bspline_handler.n_features == 23  # 20 + 3
        assert len(bspline_handler.columns) == 25  # 1 + 23 + 1

    def test_energy_features(self, unary_chemistry, simple_molecule):
        bspline_config = bspline.BSplineConfig(unary_chemistry)
        bspline_handler = BasisProcessor(unary_chemistry,
                                         bspline_config)
        vector = bspline_handler.featurize_energy_2B(simple_molecule,
                                                     simple_molecule)
        assert len(vector) == 23 + 1  # 23 features and one 1-body term

    def test_force_features(self, unary_chemistry, simple_molecule):
        bspline_config = bspline.BSplineConfig(unary_chemistry)
        bspline_handler = BasisProcessor(unary_chemistry,
                                         bspline_config)
        vector = bspline_handler.featurize_force_2B(simple_molecule,
                                                    simple_molecule)
        assert vector.shape == (3, 3, 24)  # 3 forces per atom

    def test_evaluate_single(self, unary_chemistry, simple_molecule):
        bspline_config = bspline.BSplineConfig(unary_chemistry)
        bspline_handler = BasisProcessor(unary_chemistry,
                                         bspline_config)
        eval_map = bspline_handler.evaluate_configuration(simple_molecule,
                                                          energy=1.5)
        assert len(eval_map['energy']) == 1 + 23 + 1  # number of columns
        assert eval_map['energy'][0] == 1.5  # energy value
        assert eval_map['energy'][1] == 3  # scalar for 1-body energy offset
        eval_map = bspline_handler.evaluate_configuration(simple_molecule,
                                                          name='sample',
                                                          forces=[[2, 0, 0],
                                                                  [0, 0, 0],
                                                                  [0, 0, 0]])
        assert len(eval_map) == 3 + 3 + 3  # 3 atoms, 3 forces each
        assert eval_map[('sample', 'fx_0')][0] == 2  # force value
        assert eval_map[('sample', 'fy_1')][1] == 0  # no 1-body energy offset
        assert len(eval_map[('sample', 'fz_2')]) == 1 + 23 + 1   # columns

    def test_evaluate_unary(self, unary_chemistry, simple_molecule):
        bspline_config = bspline.BSplineConfig(unary_chemistry)
        bspline_handler = BasisProcessor(unary_chemistry,
                                         bspline_config)
        df = pd.DataFrame(columns=['geometry', 'energy', 'fx', 'fy', 'fz'])
        df.loc[0] = [None,
                     1.5,
                     [4, 3, 0],
                     [0, 1, 2],
                     [2, 1, 0]]
        df.at[0, 'geometry'] = simple_molecule
        data_coordinator = io.DataCoordinator()
        df_features = bspline_handler.evaluate(df,
                                               data_coordinator,
                                               progress_bar=False)
        assert len(df_features) == 1 + 3 * 3  # energy and 3 forces per atom
        assert len(df_features.columns) == 1 + 23 + 1
        x, y, w = bspline_handler.get_training_tuples(df_features,
                                                      0.5,
                                                      data_coordinator)
        assert x.shape == (1 + 3 * 3, 23 + 1)
        assert np.allclose(y, [1.5, 4, 3, 0, 0, 1, 2, 2, 1, 0])

    def test_evaluate_binary(self, water_chemistry, simple_water):
        bspline_config = bspline.BSplineConfig(water_chemistry)
        bspline_handler = BasisProcessor(water_chemistry,
                                         bspline_config)
        df = pd.DataFrame(columns=['geometry', 'energy', 'fx', 'fy', 'fz'])
        df.loc[0] = [None,
                     1.5,
                     [4, 3, 0],
                     [0, 1, 2],
                     [2, 1, 0]]
        df.at[0, 'geometry'] = simple_water
        data_coordinator = io.DataCoordinator()
        df_feats = bspline_handler.evaluate(df,
                                            data_coordinator,
                                            progress_bar=False)
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