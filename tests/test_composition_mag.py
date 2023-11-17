import ase
import numpy as np
from uf3.data.composition import ChemicalSystem



class TestMagneticChemistryConfig:
    def test_unary(self):
        element_list = ['Fe']
        mag_element_list = ['Fe']
        handler = ChemicalSystem(element_list, 
                                 mag_element_list)
        #assert handler.interactions_map[2] == [('Fe', 'Fe')]
        assert handler.interactions_map['Magnetic_interaction'] == [('Fe', 'Fe')]
        #assert handler.numbers == [26]
        assert handler.mag_numbers == [26]

    def test_binary(self):
        element_list = ['Fe', 'C']
        mag_element_list = ['Fe']
        handler = ChemicalSystem(element_list, 
                                 mag_element_list)
        #assert len(handler.interactions_map[2]) == 3
        assert handler.interactions_map['Magnetic_interaction'] == [('Fe', 'Fe')]
        #assert handler.numbers == [6, 26]
        assert handler.mag_numbers == [26]

    def test_ternary(self):
        element_list = ['Fe', 'Mn', 'C']
        mag_element_list = ['Fe', 'Mn']
        handler = ChemicalSystem(element_list, 
                                 mag_element_list)
        #assert len(handler.interactions_map[2]) == 6
        assert handler.interactions_map['Magnetic_interaction'] == [('Fe', 'Fe'), ('Fe', 'Mn'), ('Mn', 'Mn')]
        #assert handler.numbers == [6, 25, 26]
        assert handler.mag_numbers == [25, 26]
        
    #def test_maglist(self):
        #element_list = ['Fe', 'Mn', 'C']
        #mag_element_list = ['Fe', 'Ni']
        #handler = ChemicalSystem(element_list, 
                                 #mag_element_list)
        #assert handler.maglist_check == 
        #how to test this?