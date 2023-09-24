import pytest
import numpy as np
import ase
from uf3.data import composition
from uf3.representation.process import *
from uf3.regression import optimize

@pytest.fixture()
def Nb_Sn_chemistry():
    element_list = ['Nb', 'Sn']
    chemistry_config = composition.ChemicalSystem(element_list,degree=3)
    yield chemistry_config

@pytest.fixture()
def bspline_config_larger_cutoff(Nb_Sn_chemistry):
    element_list = ['Nb', 'Sn']
    chemistry_config = composition.ChemicalSystem(element_list,degree=3)
    bspline_config = optimize.get_bspline_config(Nb_Sn_chemistry, rmin=0,
                                                    rmax_2b=6, rmax_3b=4,
                                                    knot_spacing=0.4)
    yield bspline_config


@pytest.fixture()
def bspline_config_smaller_cutoff(Nb_Sn_chemistry):
    element_list = ['Nb', 'Sn']
    chemistry_config = composition.ChemicalSystem(element_list,degree=3)
    bspline_config = optimize.get_bspline_config(Nb_Sn_chemistry, rmin=0,
                                                    rmax_2b=4.4, rmax_3b=3.2,
                                                    knot_spacing=0.4)
    yield bspline_config

@pytest.fixture()
def Nb3Sn_geom():
    geom = ase.Atoms('Nb6Sn2',
                    positions = [[0.000000000, 2.667305470, 1.333652735],
                                [0.000000000, 2.667305470, 4.000958204],
                                [1.333652735, 0.000000000, 2.667305470],
                                [4.000958204, 0.000000000, 2.667305470],
                                [2.667305470, 1.333652735, 0.000000000],
                                [2.667305470, 4.000958204, 0.000000000],
                                [0.000000000, 0.000000000, 0.000000000],
                                [2.667305470, 2.667305470, 2.667305470]],
                    pbc = True,
                    cell = [[5.3346109390, 0.0000000000, 0.0000000000],
                            [0.0000000000, 5.3346109390, 0.0000000000],
                            [0.0000000000, 0.0000000000, 5.3346109390]])

    yield geom


class TestOptimize:

    def test_get_bspline_config(self,bspline_config_larger_cutoff):

        for i in bspline_config_larger_cutoff.interactions_map[2]:
            assert bspline_config_larger_cutoff.r_min_map[i] == 0
            assert bspline_config_larger_cutoff.r_max_map[i] == 6
            assert bspline_config_larger_cutoff.resolution_map[i] == 15

        for i in bspline_config_larger_cutoff.interactions_map[3]:
            assert bspline_config_larger_cutoff.r_min_map[i] == [0,0,0]
            assert bspline_config_larger_cutoff.r_max_map[i] == [4, 4, 8]
            assert bspline_config_larger_cutoff.resolution_map[i] == [10,10,20]


    def test_get_possible_lower_cutoffs(self,bspline_config_larger_cutoff):
        cutoff_dict = optimize.get_possible_lower_cutoffs(bspline_config_larger_cutoff)
        
        assert np.allclose(cutoff_dict['rmax_2b_poss'],np.array([0.4, 0.8, 1.2,\
                1.6, 2. , 2.4, 2.8, 3.2, 3.6, 4. , 4.4, 4.8, 5.2, 5.6, 6. ]))

        assert np.allclose(cutoff_dict['rmax_3b_poss'],np.array([0.8, 1.6, 2.4, 3.2, 4. ]))

    def test_drop_columns(self,bspline_config_larger_cutoff,Nb_Sn_chemistry, Nb3Sn_geom):
        
        bspline_handler_larger_cutoff = BasisFeaturizer(bspline_config_larger_cutoff)
        
        columns_to_drop_2b = optimize.get_columns_to_drop_2b(original_bspline_config=bspline_config_larger_cutoff,
                                                     modify_2b_cutoff=4.4,
                                                     knot_spacing=0.4)

        columns_to_drop_3b = optimize.get_columns_to_drop_3b(original_bspline_config=bspline_config_larger_cutoff,
                                                     modify_3b_cutoff=3.2,
                                                     knot_spacing=0.4)

        columns_to_drop = columns_to_drop_2b + columns_to_drop_3b

        bspline_config_smaller_cutoff = optimize.get_bspline_config(Nb_Sn_chemistry, rmin=0,
                                                                    rmax_2b=4.4, rmax_3b=3.2,
                                                                    knot_spacing=0.4)
        bspline_handler_smaller_cutoff = BasisFeaturizer(bspline_config_smaller_cutoff)

        assert len(bspline_handler_larger_cutoff.columns) - len(columns_to_drop) == len(bspline_handler_smaller_cutoff.columns)

        energy_feature_larger_cutoff = bspline_handler_larger_cutoff.evaluate_configuration(geom=Nb3Sn_geom,energy=-69.44185169)
        energy_feature_smaller_cutoff = bspline_handler_smaller_cutoff.evaluate_configuration(geom=Nb3Sn_geom,energy=-69.44185169)

        index_to_drop = np.where(np.isin(bspline_handler_larger_cutoff.columns,columns_to_drop))[0]
        energy_feature_from_larger_cutoff = np.delete(energy_feature_larger_cutoff['energy'],index_to_drop)

        assert np.allclose(energy_feature_smaller_cutoff['energy'],energy_feature_from_larger_cutoff)

