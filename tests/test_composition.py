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

    def test_quaternary(self):
        element_list = ['He', 'Li', 'H', 'Be']  # out of order on purpose
        handler = ChemicalSystem(element_list, degree=3)
        assert handler.interactions_map[2] == [
            ('H', 'H'), ('H', 'He'),  ('H', 'Li'),  ('H', 'Be' ),
                        ('He', 'He'), ('He', 'Li'), ('He', 'Be'),
                                      ('Li', 'Li'), ('Li', 'Be'),
                                                    ('Be', 'Be')
                                                    ]  # in this order
        assert handler.interactions_map[3] == [
            ('H', 'H', 'H'  ), ('H', 'H', 'He'), ('H', 'H', 'Li'), ('H', 'H', 'Be'),
            ('H', 'He', 'He'), ('H', 'He', 'Li'), ('H', 'He', 'Be'),
            ('H', 'Li', 'Li'), ('H', 'Li', 'Be'),
            ('H', 'Be', 'Be'),
            ('He', 'H', 'H' ), ('He', 'H', 'He'), ('He', 'H', 'Li'), ('He', 'H', 'Be'),
            ('He', 'He', 'He'), ('He', 'He', 'Li'), ('He', 'He', 'Be'),
            ('He', 'Li', 'Li'), ('He', 'Li', 'Be'),
            ('He', 'Be', 'Be'),
            ('Li', 'H', 'H' ), ('Li', 'H', 'He'), ('Li', 'H', 'Li'), ('Li', 'H', 'Be'),
            ('Li', 'He', 'He'), ('Li', 'He', 'Li'), ('Li', 'He', 'Be'),
            ('Li', 'Li', 'Li'), ('Li', 'Li', 'Be'),
            ('Li', 'Be', 'Be'),
            ('Be', 'H', 'H' ), ('Be', 'H', 'He'), ('Be', 'H', 'Li'), ('Be', 'H', 'Be'),
            ('Be', 'He', 'He'), ('Be', 'He', 'Li'), ('Be', 'He', 'Be'),
            ('Be', 'Li', 'Li'), ('Be', 'Li', 'Be'),
            ('Be', 'Be', 'Be')
        ]  # in this order
        assert handler.numbers == [1, 2, 3, 4,]

    def test_remove_duplicates(self):
        element_list = ['H', 'H', 'He']
        handler = ChemicalSystem(element_list)
        assert handler.numbers == [1, 2]

    def test_composition_tuple(self):
        element_list = ['Al', 'Cu', 'Zr']
        handler = ChemicalSystem(element_list)
        geom = ase.Atoms('Al2Zr5')
        comp = handler.get_composition_tuple(geom)
        assert np.allclose(comp, [2, 0, 5])
