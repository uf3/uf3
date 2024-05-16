import pytest
from uf3.data.geometry import *
from uf3.representation.distances import *
import itertools



@pytest.fixture()
def simple_molecule():
    geom = ase.Atoms('Fe3',
                     positions=[[0, 0, 0], [3, 0.0, 0.0], [0, 4.0, 0]],
                     pbc=False,
                     cell=None)
    yield geom


@pytest.fixture()
def simple_unary():
    geom = ase.Atoms('Fe4', 
                     positions=[[0, 0, 0],[0, 1.25, 1.75], [1.25, 0, 1.75], [1.25, 1.25, 0]], 
                     pbc=True, 
                     cell=[[2.5, 0, 0], [0, 2.5, 0], [0, 0, 3.5]])
    yield geom


@pytest.fixture()
def simple_binary():
    geom = ase.Atoms('FeMn',
                     positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
                     pbc=True,
                     cell=[[2.5, 0, 0], [0, 2.5, 0], [0, 0, 3.5]])
    yield geom
    
    
    
class TestMagneticMolecule:
    def test_distances_and_magmom(self, simple_molecule):
        geom = simple_molecule
        # parameters
        mag_element_list = ['Fe']
        magmom_list = [2.2]
        cwr = itertools.combinations_with_replacement(mag_element_list, 2)
        mag_tuples = sorted(cwr)
        magnetic_r_min_map = {('Fe', 'Fe'): 0.5}
        magnetic_r_max_map = {('Fe', 'Fe'): 6.5}
        distances = distances_by_interaction(geom,
                                             mag_tuples,
                                             magnetic_r_min_map,
                                             magnetic_r_max_map)
        # get_magmom_matrix test (AFM Fe4)
        magmom_matrix = get_magmom_matrix(geom,
                                          geom,
                                          magmom_list
                                          )
        # magmom_by_interaction test (AFM Fe4)
        magmom_map = magmom_by_interaction(geom,
                                           mag_tuples,
                                           magnetic_r_min_map,
                                           magnetic_r_max_map,
                                           magmom_list
                                           )
        assert len(distances['Fe', 'Fe']) == 6
        assert len(magmom_map['Fe', 'Fe']) == 6
        assert magmom_matrix.shape == (3, 3)
        assert magmom_matrix[2][-1] == [2.2, 2.2]
    def test_distances_derivatives_and_magmom(self, simple_molecule):
        geom = simple_molecule
        # parameters
        mag_element_list = ['Fe']
        magmom_list = [2.2]
        cwr = itertools.combinations_with_replacement(mag_element_list, 2)
        mag_tuples = sorted(cwr)
        magnetic_r_min_map = {('Fe', 'Fe'): 0.5}
        magnetic_r_max_map = {('Fe', 'Fe'): 6.5}
        # derivatives_by_interaction_and_mag_map test
        r_cut = 6.5
        distances, derivatives, magmom_map = derivatives_by_interaction_and_mag_map(geom,
                                                                                    mag_tuples,
                                                                                    magmom_list,
                                                                                    r_cut,
                                                                                    magnetic_r_min_map,
                                                                                    magnetic_r_max_map,
                                                                                    )
        assert len(distances['Fe', 'Fe']) == 6
        assert derivatives['Fe', 'Fe'].shape == (3, 3, 6)
        assert len(magmom_map['Fe', 'Fe']) == 6


class TestMagneticUnary:
    def test_distances_and_magmom(self, simple_unary):
        geom = simple_unary
        supercell = get_supercell(geom, r_cut=2)
        # parameters
        mag_element_list = ['Fe']
        magmom_list = [2.2, -2.2, -2.2, 2.2]
        cwr = itertools.combinations_with_replacement(mag_element_list, 2)
        mag_tuples = sorted(cwr)
        magnetic_r_min_map = {('Fe', 'Fe'): 0.5}
        magnetic_r_max_map = {('Fe', 'Fe'): 3.5}
        distances = distances_by_interaction(geom,
                                             mag_tuples,
                                             magnetic_r_min_map,
                                             magnetic_r_max_map,
                                             supercell)
        # get_magmom_matrix test (AFM Fe4)
        magmom_matrix = get_magmom_matrix(geom,
                                          supercell,
                                          magmom_list
                                          )
        # magmom_by_interaction test (AFM Fe4)
        magmom_map = magmom_by_interaction(geom,
                                           mag_tuples,
                                           magnetic_r_min_map,
                                           magnetic_r_max_map,
                                           magmom_list,
                                           supercell,
                                           )
        assert len(distances['Fe', 'Fe']) == 536
        assert len(magmom_map['Fe', 'Fe']) == 536
        assert magmom_matrix.shape == (4, 108)
        assert magmom_matrix[2][-1] == [-2.2, 2.2]
    def test_distances_derivatives_and_magmom(self, simple_unary):
        geom = simple_unary
        supercell = get_supercell(geom, r_cut=2)
        # parameters
        mag_element_list = ['Fe']
        magmom_list = [2.2, -2.2, -2.2, 2.2]
        cwr = itertools.combinations_with_replacement(mag_element_list, 2)
        mag_tuples = sorted(cwr)
        magnetic_r_min_map = {('Fe', 'Fe'): 0.5}
        magnetic_r_max_map = {('Fe', 'Fe'): 3.5}
        # derivatives_by_interaction_and_mag_map test
        r_cut = 3.0
        distances, derivatives, magmom_map = derivatives_by_interaction_and_mag_map(geom,
                                                                                    mag_tuples,
                                                                                    magmom_list,
                                                                                    r_cut,
                                                                                    magnetic_r_min_map,
                                                                                    magnetic_r_max_map,
                                                                                    supercell,
                                                                                    )
        assert len(distances['Fe', 'Fe']) == 536
        assert derivatives['Fe', 'Fe'].shape == (4, 3, 536)
        assert len(magmom_map['Fe', 'Fe']) == 536
        
        
        
class TestMagneticBinary:
    def test_distances_and_magmom(self, simple_binary):
        geom = simple_binary
        supercell = get_supercell(geom, r_cut=2)
        # parameters
        mag_element_list = ['Fe', 'Mn']
        magmom_list = [2.2, 2.7]
        cwr = itertools.combinations_with_replacement(mag_element_list, 2)
        mag_tuples = sorted(cwr)
        magnetic_r_min_map = {('Fe', 'Fe'): 0.5, ('Fe', 'Mn'): 0.5, ('Mn', 'Mn'): 0.5}
        magnetic_r_max_map = {('Fe', 'Fe'): 6.5, ('Fe', 'Mn'): 3.5, ('Mn', 'Mn'): 4.5}
        distances = distances_by_interaction(geom,
                                             mag_tuples,
                                             magnetic_r_min_map,
                                             magnetic_r_max_map,
                                             supercell)
        # get_magmom_matrix test (AFM Fe4)
        magmom_matrix = get_magmom_matrix(geom,
                                          supercell,
                                          magmom_list
                                          )
        # magmom_by_interaction test (AFM Fe4)
        magmom_map = magmom_by_interaction(geom,
                                           mag_tuples,
                                           magnetic_r_min_map,
                                           magnetic_r_max_map,
                                           magmom_list,
                                           supercell,
                                           )
        assert len(distances['Fe', 'Fe']) == 30
        assert len(magmom_map['Fe', 'Fe']) == 30
        assert len(distances['Fe', 'Mn']) == 42
        assert len(magmom_map['Fe', 'Mn']) == 42
        assert len(distances['Mn', 'Mn']) == 22
        assert len(magmom_map['Mn', 'Mn']) == 22
        assert magmom_matrix.shape == (2, 54)
        assert magmom_matrix[1][-2] == [2.7, 2.2]
    def test_distances_derivatives_and_magmom(self, simple_binary):
        geom = simple_binary
        supercell = get_supercell(geom, r_cut=2)
        # parameters
        mag_element_list = ['Fe', 'Mn']
        magmom_list = [2.2, 2.7]
        cwr = itertools.combinations_with_replacement(mag_element_list, 2)
        mag_tuples = sorted(cwr)
        magnetic_r_min_map = {('Fe', 'Fe'): 0.5, ('Fe', 'Mn'): 0.5, ('Mn', 'Mn'): 0.5}
        magnetic_r_max_map = {('Fe', 'Fe'): 6.5, ('Fe', 'Mn'): 3.5, ('Mn', 'Mn'): 4.5}
        # derivatives_by_interaction_and_mag_map test
        r_cut = 3.0
        distances, derivatives, magmom_map = derivatives_by_interaction_and_mag_map(geom,
                                                                                    mag_tuples,
                                                                                    magmom_list,
                                                                                    r_cut,
                                                                                    magnetic_r_min_map,
                                                                                    magnetic_r_max_map,
                                                                                    supercell,
                                                                                    )
        assert len(distances['Fe', 'Fe']) == 30
        assert len(magmom_map['Fe', 'Fe']) == 30
        assert len(distances['Fe', 'Mn']) == 42
        assert len(magmom_map['Fe', 'Mn']) == 42
        assert len(distances['Mn', 'Mn']) == 22
        assert len(magmom_map['Mn', 'Mn']) == 22
        assert derivatives['Fe', 'Fe'].shape == (2, 3, 30)
        assert derivatives['Fe', 'Mn'].shape == (2, 3, 42)
        assert derivatives['Mn', 'Mn'].shape == (2, 3, 22)
    
