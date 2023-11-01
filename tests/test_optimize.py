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
def config_1():
    min_max_ks = {}
    min_max_ks["rmin_2b"] = 0.01
    min_max_ks["rmax_2b"] = 6.01
    min_max_ks["rmin_3b"] = 0.8
    min_max_ks["rmax_3b"] = 4
    min_max_ks["knot_spacing_2b"] = 0.4
    min_max_ks["knot_spacing_3b"] = 0.8
    yield min_max_ks


@pytest.fixture()
def bspline_config_larger_cutoff(Nb_Sn_chemistry,config_1):
    element_list = ['Nb', 'Sn']
    chemistry_config = composition.ChemicalSystem(element_list,degree=3)
    bspline_config = optimize.get_bspline_config(Nb_Sn_chemistry,
                                                    rmin_2b=config_1["rmin_2b"],
                                                    rmin_3b=config_1["rmin_3b"],
                                                    rmax_2b=config_1["rmax_2b"],
                                                    rmax_3b=config_1["rmax_3b"],
                                                    knot_spacing_2b=config_1["knot_spacing_2b"],
                                                    knot_spacing_3b=config_1["knot_spacing_3b"],
                                                    leading_trim=0,
                                                    trailing_trim=3)
    yield bspline_config

@pytest.fixture()
def config_2():
    min_max_ks = {}
    min_max_ks["rmin_2b"] = 0.1
    min_max_ks["rmax_2b"] = 9.1
    min_max_ks["rmin_3b"] = 0.9
    min_max_ks["rmax_3b"] = 7.2
    min_max_ks["knot_spacing_2b"] = 0.3
    min_max_ks["knot_spacing_3b"] = 0.9
    yield min_max_ks

@pytest.fixture()
def bspline_config_larger_cutoff_2(Nb_Sn_chemistry,config_2):
    element_list = ['Nb', 'Sn']
    chemistry_config = composition.ChemicalSystem(element_list,degree=3)
    bspline_config = optimize.get_bspline_config(Nb_Sn_chemistry,
                                                    rmin_2b=config_2["rmin_2b"],
                                                    rmin_3b=config_2["rmin_3b"],
                                                    rmax_2b=config_2["rmax_2b"],
                                                    rmax_3b=config_2["rmax_3b"],
                                                    knot_spacing_2b=config_2["knot_spacing_2b"],
                                                    knot_spacing_3b=config_2["knot_spacing_3b"],
                                                    leading_trim=0,
                                                    trailing_trim=3)
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
                    cell = [[6.3346109390, 0.0000000000, 0.0000000000],
                            [0.0000000000, 6.3346109390, 0.0000000000],
                            [0.0000000000, 0.0000000000, 6.3346109390]])

    yield geom


class TestOptimize:

    def test_drop_columns(self,bspline_config_larger_cutoff,Nb_Sn_chemistry, Nb3Sn_geom, config_1):
        
        cutoff_dict = optimize.get_lower_cutoffs(bspline_config_larger_cutoff)
        
        for i in range(len(cutoff_dict['lower_rmax_2b'])):
            for j in range(len(cutoff_dict['lower_rmax_3b'])):
                bspline_config_smaller_cutoff = optimize.get_bspline_config(Nb_Sn_chemistry,
                                                                        rmin_2b=config_1["rmin_2b"],
                                                                        rmin_3b=config_1["rmin_3b"],
                                                                        rmax_2b=cutoff_dict['lower_rmax_2b'][i],
                                                                        rmax_3b=cutoff_dict['lower_rmax_3b'][j],
                                                                        knot_spacing_2b=config_1["knot_spacing_2b"],
                                                                        knot_spacing_3b=config_1["knot_spacing_3b"],
                                                                        leading_trim=0,
                                                                        trailing_trim=3)

                bspline_handler_larger_cutoff = BasisFeaturizer(bspline_config_larger_cutoff)
                bspline_handler_smaller_cutoff = BasisFeaturizer(bspline_config_smaller_cutoff)
        
                columns_to_drop_2b = optimize.get_columns_to_drop_2b(original_bspline_config=bspline_config_larger_cutoff,
                                                                    modify_2b_cutoff=cutoff_dict['lower_rmax_2b'][i],
                                                                    knot_spacing_2b=config_1["knot_spacing_2b"])

                columns_to_drop_3b = optimize.get_columns_to_drop_3b(original_bspline_config=bspline_config_larger_cutoff,
                                                                    modify_3b_cutoff=cutoff_dict['lower_rmax_3b'][j],
                                                                    knot_spacing_3b=config_1["knot_spacing_3b"])

                columns_to_drop = columns_to_drop_2b + columns_to_drop_3b

                assert len(bspline_handler_larger_cutoff.columns) - len(columns_to_drop) == len(bspline_handler_smaller_cutoff.columns)

                energy_feature_larger_cutoff = bspline_handler_larger_cutoff.evaluate_configuration(geom=Nb3Sn_geom,energy=-69.44185169)
                energy_feature_smaller_cutoff = bspline_handler_smaller_cutoff.evaluate_configuration(geom=Nb3Sn_geom,energy=-69.44185169)

                index_to_drop = np.where(np.isin(bspline_handler_larger_cutoff.columns,columns_to_drop))[0]
                energy_feature_from_larger_cutoff = np.delete(energy_feature_larger_cutoff['energy'],index_to_drop)

                assert np.allclose(energy_feature_smaller_cutoff['energy'],energy_feature_from_larger_cutoff)


    def test_drop_columns_2(self,bspline_config_larger_cutoff_2,Nb_Sn_chemistry, Nb3Sn_geom, config_2):
        
        cutoff_dict = optimize.get_lower_cutoffs(bspline_config_larger_cutoff_2)
        for i in range(len(cutoff_dict['lower_rmax_2b'])):
            for j in range(len(cutoff_dict['lower_rmax_3b'])):
                bspline_config_smaller_cutoff = optimize.get_bspline_config(Nb_Sn_chemistry,
                                                                        rmin_2b=config_2["rmin_2b"],
                                                                        rmin_3b=config_2["rmin_3b"],
                                                                        rmax_2b=cutoff_dict['lower_rmax_2b'][i],
                                                                        rmax_3b=cutoff_dict['lower_rmax_3b'][j],
                                                                        knot_spacing_2b=config_2["knot_spacing_2b"],
                                                                        knot_spacing_3b=config_2["knot_spacing_3b"],
                                                                        leading_trim=0,
                                                                        trailing_trim=3)


                bspline_handler_larger_cutoff = BasisFeaturizer(bspline_config_larger_cutoff_2)
                bspline_handler_smaller_cutoff = BasisFeaturizer(bspline_config_smaller_cutoff)
        
                columns_to_drop_2b = optimize.get_columns_to_drop_2b(original_bspline_config=bspline_config_larger_cutoff_2,
                                                                    modify_2b_cutoff=cutoff_dict['lower_rmax_2b'][i],
                                                                    knot_spacing_2b=config_2["knot_spacing_2b"])

                columns_to_drop_3b = optimize.get_columns_to_drop_3b(original_bspline_config=bspline_config_larger_cutoff_2,
                                                                    modify_3b_cutoff=cutoff_dict['lower_rmax_3b'][j],
                                                                    knot_spacing_3b=config_2["knot_spacing_3b"])

                columns_to_drop = columns_to_drop_2b + columns_to_drop_3b

                assert len(bspline_handler_larger_cutoff.columns) - len(columns_to_drop) == len(bspline_handler_smaller_cutoff.columns)

                energy_feature_larger_cutoff = bspline_handler_larger_cutoff.evaluate_configuration(geom=Nb3Sn_geom,energy=-69.44185169)
                energy_feature_smaller_cutoff = bspline_handler_smaller_cutoff.evaluate_configuration(geom=Nb3Sn_geom,energy=-69.44185169)

                index_to_drop = np.where(np.isin(bspline_handler_larger_cutoff.columns,columns_to_drop))[0]
                energy_feature_from_larger_cutoff = np.delete(energy_feature_larger_cutoff['energy'],index_to_drop)

                assert np.allclose(energy_feature_smaller_cutoff['energy'],energy_feature_from_larger_cutoff)

