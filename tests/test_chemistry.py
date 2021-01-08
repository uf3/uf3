import ase
import numpy as np
from uf3.data.composition import ChemicalSystem


class TestChemistryConfig:
    def test_unary(self):
        element_list = ['Au']
        handler = ChemicalSystem(element_list,
                                 r_min_map={('Au', 'Au'): 1.1})
        assert handler.interactions_map[2] == [('Au', 'Au')]
        assert handler.r_min_map[('Au', 'Au')] == 1.1
        assert handler.r_max_map[('Au', 'Au')] == 6.0
        assert handler.resolution_map[('Au', 'Au')] == 20
        assert handler.numbers == [79]

    def test_binary(self):
        element_list = ['Ne', 'Xe']
        handler = ChemicalSystem(element_list,
                                 resolution_map={('Ne', 'Xe'): 10})
        assert len(handler.interactions_map[2]) == 3
        assert handler.r_min_map[('Ne', 'Ne')] == 0.0
        assert handler.r_max_map[('Xe', 'Xe')] == 6.0
        assert handler.resolution_map[('Ne', 'Xe')] == 10
        assert handler.numbers == [10, 54]

    def test_ternary(self):
        element_list = ['Al', 'Cu', 'Zr']
        handler = ChemicalSystem(element_list)
        assert len(handler.interactions_map[2]) == 6
        print(handler.interactions_map)
        assert handler.r_min_map[('Al', 'Zr')] == 0.0
        assert handler.r_max_map[('Cu', 'Zr')] == 6.0
        assert handler.resolution_map[('Al', 'Cu')] == 20
        assert handler.numbers == [13, 29, 40]

    def test_composition_tuple(self):
        element_list = ['Al', 'Cu', 'Zr']
        handler = ChemicalSystem(element_list)
        geom = ase.Atoms('Al2Zr5')
        comp = handler.get_composition_tuple(geom)
        assert np.allclose(comp, [2, 0, 5])
