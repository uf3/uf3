import ase
import numpy as np
from uf3.data.composition import ChemicalSystem


class TestChemistryConfig:
    def test_unary(self):
        element_list = ['Au']
        handler = ChemicalSystem(element_list)
        assert handler.interactions_map[2] == [('Au', 'Au')]
        assert handler.numbers == [79]

    def test_binary(self):
        element_list = ['Ne', 'Xe']
        handler = ChemicalSystem(element_list,
                                 )
        assert len(handler.interactions_map[2]) == 3
        assert handler.numbers == [10, 54]

    def test_ternary(self):
        element_list = ['Al', 'Cu', 'Zr']
        handler = ChemicalSystem(element_list)
        assert len(handler.interactions_map[2]) == 6
        assert handler.numbers == [13, 29, 40]

    def test_composition_tuple(self):
        element_list = ['Al', 'Cu', 'Zr']
        handler = ChemicalSystem(element_list)
        geom = ase.Atoms('Al2Zr5')
        comp = handler.get_composition_tuple(geom)
        assert np.allclose(comp, [2, 0, 5])
