import pytest
import ase
from uf3.representation.process import *
from uf3.representation import bspline
from uf3.data import composition
from uf3.data import io
import numpy as np

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
def simple_water_2():
    geom = ase.Atoms('H2O',
                     positions=[[0, 0, 0], [1.5, 0.0, 0.0], [0, 2.0, 0]],
                     pbc=False,
                     cell=None)
    yield geom

@pytest.fixture()
def simple_molecule_CPtC():
    geom = ase.Atoms('CPtC',
                      positions = [[0., 0., 0.], [0., 1.5, 0.], [0., 0., 2.]],
                      pbc = True,
                      cell = [30.0, 30.0, 30.0])
    yield geom

@pytest.fixture()
def atoms_molecule_CCPt():
    geom = ase.Atoms('C2Pt',
                      positions = [[0., 0., 0.], [0., 0., 2.], [0., 1.5, 0.]],
                      pbc = True,
                      cell = [30.0, 30.0, 30.0])
    yield geom
    
@pytest.fixture()    
def atoms_molecule_Yb2La2():
    geom = ase.Atoms('Yb2La2',
                      positions = [[0., 0., 0.], [0., 0., 2.], [0., 1.5, 0.],[2.,0,0]],
                      pbc = True,
                      cell = [30.0, 30.0, 30.0]) 
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
    
@pytest.fixture()
def binary_chemistry_equal_electronegativity():
    element_list = ['Yb', 'La']
    chemistry_config_1 = composition.ChemicalSystem(element_list,degree=3)
    element_list = ['La', 'Yb']
    chemistry_config_2 = composition.ChemicalSystem(element_list,degree=3)    
    yield [chemistry_config_1,chemistry_config_2]
    
@pytest.fixture()
def binary_chemistry_3B():
    element_list = ['C', 'Pt']
    chemistry_config = composition.ChemicalSystem(element_list,degree=3)
    yield chemistry_config

@pytest.fixture()
def water_2_feature():
    two_body = {('H', 'H'): np.array([0.0, 0.40032798833819255, 1.1900510204081631, 
                    0.40949951409135077, 0.00012147716229348758, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                ('H', 'O'): np.array([0.0, 0.0, 0.20991253644314867, 1.4571185617103986, 
                    1.745019436345967, 0.5846695821185617, 0.0032798833819242057, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                ('O', 'O'): np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
    three_body = {('H', 'H', 'H'):{"position":np.array([]),"value":np.array([])}, 
                ('H', 'H', 'O'):{"position":np.array([9, 10, 11, 12, 16, 17, 18, 
                    19, 20, 24, 25, 26, 27, 28, 33, 34, 35, 36, 37, 52, 53, 54, 
                    55, 59, 60, 61, 62, 63, 68, 69, 70, 71, 72, 78, 79, 80, 81, 
                    82, 97, 98, 99, 100, 105, 106, 107, 108, 109, 115, 116, 117, 
                    118, 119, 125, 126, 127, 128, 129, 145, 146, 147, 148, 154, 
                    155, 156, 157, 158, 164, 165, 166, 167, 168, 174, 175, 176, 
                    177, 178]),
                    "value":np.array([8.998308318036238e-06, 5.553122679602926e-05, 
                        2.1162688081307482e-05, 4.16588348057235e-08, 8.998308318036216e-06, 
                        0.0004546957671957677, 0.0022863479443642424, 0.0008419840681063094, 
                        1.652467113960362e-06, 0.0002069610913148332, 0.002806441139065861, 
                        0.005365949534820799, 0.001370294467973358, 2.5828477579548518e-06, 
                        0.0002069610913148332, 0.0022965370010438087, 0.0022191800163791483, 
                        0.00017107547669926987, 2.2218045229719143e-07, 0.0007536083216355335, 
                        0.004650740244167441, 0.0017723751268094983, 3.4889274149793364e-06, 
                        0.0007536083216355316, 0.038080770502645474, 0.1914816403405049, 
                        0.07051616570390326, 0.00013839412079418004, 0.017332991397617244, 
                        0.23503944539676538, 0.44939827354124107, 0.11476216169276852, 
                        0.00021631349972871838, 0.017332991397617244, 0.19233497383741857, 
                        0.1858563263717533, 0.01432757117356382, 1.8607612879889746e-05, 
                        0.0009935632101164991, 0.006131572958728215, 0.0023367134756443622, 
                        4.599829676465292e-06, 0.0009935632101164967, 0.05020599096119924, 
                        0.2524509188568845, 0.09296907418673811, 0.00018245991049978955, 
                        0.022851953832679444, 0.3098778757718547, 0.5924902611364619, 
                        0.15130334750539126, 0.0002851894399408475, 0.022851953832679444, 
                        0.2535759605319199, 0.2450344601418637, 0.018889583885544334, 
                        2.453242494114816e-05, 0.00018746475662575454, 0.0011569005582506068, 
                        0.00044088933502723824, 8.678923917859043e-07, 0.00018746475662575405, 
                        0.00947282848324514, 0.04763224884092161, 0.01754133475221474, 
                        3.442639820750746e-05, 0.004311689402392347, 0.05846752373053862, 
                        0.11179061530876638, 0.02854780141611156, 5.380932829072594e-05, 
                        0.004311689402392347, 0.047844520855079224, 0.046232917007898805, 
                        0.00356407243123478, 4.6287594228581435e-06])},
                ('H', 'O', 'O'):{"position":np.array([]),"value":np.array([])},
                ('O', 'H', 'H'):{"position":np.array([50, 51, 52, 53, 59, 60, 61, 
                    62, 69, 70, 71, 72, 79, 80, 81, 82, 89, 90, 91, 92, 99, 100, 
                    101, 102, 109, 110, 111, 112, 119, 120, 121, 122, 129, 130, 
                    131, 132]),
                    "value":np.array([8.998308318036216e-06, 0.00018699609473419023,
                        0.00016637497150535723, 2.343309457821933e-05, 0.0002069610913148332,
                        0.0043009101788863795, 0.0038266243446232195, 0.0005389611752990452,
                        0.0002069610913148332, 0.0043009101788863795, 0.0038266243446232195,
                        0.0005389611752990452, 0.00035693289661543583, 0.007417511757789529,
                        0.006599540536379156, 0.0009295127516026982, 0.008767351737873276,
                        0.1821965283026791, 0.16210468057005284, 0.02283164515071167,
                        0.008257447599851224, 0.17160008293440834, 0.15267676551808262,
                        0.021503769791279246, 0.01283158766151963, 0.26665643109095494,
                        0.2372507093666391, 0.033415592868540726, 0.01393538014853207,
                        0.28959461871168224, 0.2576593725379629, 0.03629005247013563,
                        0.0011037924870124407, 0.02293818762072729, 0.020408663171323782,
                        0.0028744596015948995])},
                    ('O', 'H', 'O'):{"position":np.array([]),"value":np.array([])},
                    ('O', 'O', 'O'):{"position":np.array([]),"value":np.array([])}
                }
    features = {2:two_body,3:three_body}
    yield features
    
class TestBasis:
    
    def test_equal_electronegativity(self,binary_chemistry_equal_electronegativity, atoms_molecule_Yb2La2):
        bspline_config = bspline.BSplineBasis(binary_chemistry_equal_electronegativity[0])
        bspline_handler = BasisFeaturizer(bspline_config) 
        feature_1 = bspline_handler.featurize_energy_3B(atoms_molecule_Yb2La2)
        
        bspline_config = bspline.BSplineBasis(binary_chemistry_equal_electronegativity[1])
        bspline_handler = BasisFeaturizer(bspline_config) 
        feature_2 = bspline_handler.featurize_energy_3B(atoms_molecule_Yb2La2)  
        assert np.allclose(feature_1,feature_2)
        
    def test_atom_swap_3B(self,binary_chemistry_3B,simple_molecule_CPtC,atoms_molecule_CCPt):
        bspline_config = bspline.BSplineBasis(binary_chemistry_3B)
        bspline_handler = BasisFeaturizer(bspline_config) 
        feature_1 = bspline_handler.featurize_energy_3B(simple_molecule_CPtC)
        feature_1 = feature_1[np.where(feature_1!=0)[0]]
        feature_2 = bspline_handler.featurize_energy_3B(atoms_molecule_CCPt)
        feature_2 = feature_2[np.where(feature_2!=0)[0]]
        assert np.allclose(feature_1,feature_2)
        
    def test_setup(self, unary_chemistry):
        bspline_config = bspline.BSplineBasis(unary_chemistry)
        bspline_handler = BasisFeaturizer(bspline_config)
        assert bspline_handler.r_cut == 8.0
        assert len(bspline_handler.knots_map[('Ar', 'Ar')]) == 22
        assert len(bspline_handler.basis_functions[('Ar', 'Ar')]) == 18
        assert len(bspline_handler.columns) == 20  # 1 + 23 + 1

    def test_energy_features(self, unary_chemistry, simple_molecule):
        bspline_config = bspline.BSplineBasis(unary_chemistry)
        bspline_handler = BasisFeaturizer(bspline_config)
        vector = bspline_handler.featurize_energy_2B(simple_molecule,
                                                     simple_molecule)
        assert len(vector) == 18  # 23 features

    def test_energy_features_2(self, simple_water_2, water_2_feature):
        element_list = ['H', 'O']
        chemistry_config = composition.ChemicalSystem(element_list,degree=3)
        bspline_config = bspline.BSplineBasis(chemistry_config)
        bspline_handler = BasisFeaturizer(bspline_config)
        
        features_2B = bspline_handler.featurize_energy_2B(simple_water_2)
        features_3B = bspline_handler.featurize_energy_3B(simple_water_2)
        features_con = np.concatenate((features_2B,features_3B))

        features = {}
        features = {key:[] for key in bspline_config.interactions_map[2]}
        features.update({key:[] for key in bspline_config.interactions_map[3]})
        
        for i in features.keys():
            start = bspline_config.get_interaction_partitions()[1][i]-2
            end = bspline_config.get_interaction_partitions()[1][i]-2 + \
                    bspline_config.get_interaction_partitions()[0][i]
            features[i] = features_con[start:end]

        for i in bspline_config.interactions_map[2]:
            assert np.allclose(water_2_feature[2][i],features[i])

        for i in bspline_config.interactions_map[3]:
            print(i)
            feature = features[i]
            feature_position = np.where(feature!=0)[0]
            feature_value = feature[feature_position]

            assert np.allclose(water_2_feature[3][i]["position"],feature_position)
            #In my hard-coded features I double counted every interaction,
            #so it needs to be divided by two
            assert np.allclose(water_2_feature[3][i]["value"]/2,feature_value)

        

    def test_force_features(self, unary_chemistry, simple_molecule):
        bspline_config = bspline.BSplineBasis(unary_chemistry)
        bspline_handler = BasisFeaturizer(bspline_config)
        vector = bspline_handler.featurize_force_2B(simple_molecule,
                                                    simple_molecule)
        assert vector.shape == (3, 3, 18)  # 3 forces per atom

    def test_evaluate_single(self, unary_chemistry, simple_molecule):
        bspline_config = bspline.BSplineBasis(unary_chemistry)
        bspline_handler = BasisFeaturizer(bspline_config)
        eval_map = bspline_handler.evaluate_configuration(simple_molecule,
                                                          energy=1.5)
        assert len(eval_map['energy']) == 1 + 18 + 1  # number of columns
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
        assert len(eval_map[('sample', 'fz_2')]) == 1 + 18 + 1   # columns

    def test_evaluate_unary(self, unary_chemistry, simple_molecule):
        bspline_config = bspline.BSplineBasis(unary_chemistry)
        bspline_handler = BasisFeaturizer(bspline_config)
        df = pd.DataFrame(columns=['geometry', 'energy', 'fx', 'fy', 'fz'])
        df.loc[0] = [None,
                     1.5,
                     [4, 3, 0],
                     [0, 1, 2],
                     [2, 1, 0]]
        df.at[0, 'geometry'] = simple_molecule
        df.loc[1] = [None,
                     1.5,
                     [4.1, 3.1, 0],
                     [0, 1.1, 2.1],
                     [2, 1, 0]]
        df.at[1, 'geometry'] = simple_molecule
        data_coordinator = io.DataCoordinator()
        atoms_key = data_coordinator.atoms_key
        energy_key = data_coordinator.energy_key
        df_features = bspline_handler.evaluate(df,
                                               atoms_key,
                                               energy_key,
                                               progress=False)
        assert len(df_features) == 2 + 6 * 3  # energy and 3 forces per atom
        assert len(df_features.columns) == 1 + 18 + 1
        x, y, w = bspline_handler.get_training_tuples(df_features,
                                                      0.5,
                                                      data_coordinator)
        assert x.shape == (2 + 6 * 3, 18 + 1)
        assert np.allclose(y[:10], [1.5, 4, 3, 0, 0, 1, 2, 2, 1, 0])

    def test_evaluate_binary(self, water_chemistry, simple_water):
        bspline_config = bspline.BSplineBasis(water_chemistry)
        bspline_handler = BasisFeaturizer(bspline_config)
        df = pd.DataFrame(columns=['geometry', 'energy', 'fx', 'fy', 'fz'])
        df.loc[0] = [None,
                     1.5,
                     [4, 3, 0],
                     [0, 1, 2],
                     [2, 1, 0]]
        df.at[0, 'geometry'] = simple_water
        df.loc[1] = [None,
                     1.5,
                     [4.1, 3.1, 0],
                     [0, 1.1, 2.1],
                     [2, 1, 0]]
        df.at[1, 'geometry'] = simple_water
        data_coordinator = io.DataCoordinator()
        print(len(bspline_handler.columns))
        atoms_key = data_coordinator.atoms_key
        energy_key = data_coordinator.energy_key
        df_feats = bspline_handler.evaluate(df,
                                            atoms_key,
                                            energy_key,
                                            progress=False)
        assert len(df_feats) == 2 * (1 + 3 * 3)  # energy and 3 forces per atom
        assert len(df_feats.columns) == 1 + 2 + 18 * 3
        # energy, 23 features per interaction, two 1-body terms
        x, y, w = bspline_handler.get_training_tuples(df_feats,
                                                      0.5,
                                                      data_coordinator)
        assert x.shape == (2 + 6 * 3, 18 * 3 + 2)
        assert np.allclose(y[:10], [1.5, 4, 3, 0, 0, 1, 2, 2, 1, 0])


def test_flatten_by_interactions():
    vector_map = {('A', 'A'): np.array([1, 1, 1]),
                  ('A', 'B'): np.array([2, 2]),
                  ('B', 'B'): np.array([3, 3, 3, 3])}
    pair_tuples = [('A', 'A'), ('A', 'B'), ('B', 'B')]
    vector = flatten_by_interactions(vector_map, pair_tuples)
    assert np.allclose(vector, [1, 1, 1, 2, 2, 3, 3, 3, 3])

