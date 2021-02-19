import itertools
from ase import symbols as ase_symbols
import numpy as np


class ChemicalSystem:
    """
    -Manage parameters related to composition
    -e.g. pair combinations of a binary/ternary system
    """
    def __init__(self,
                 element_list,
                 degree=2):
        """
        Args:
            element_list (list): set of elements in chemical system
                e.g. ['Ne', 'Xe'] or [10, 54]
            degree (int): handle N-body interactions
                e.g. 2 to fit pair potentials.
        """
        numbers = {el: ase_symbols.symbols2numbers(el) for el in element_list}
        self.element_list = sorted(element_list, key=lambda el: numbers[el])
        self.numbers = [numbers[el] for el in self.element_list]
        self.degree = degree
        self.interactions_map = {}
        for d in range(2, degree+1):
            cwr = itertools.combinations_with_replacement(self.element_list, d)
            self.interactions_map[d] = sorted(cwr)

    @staticmethod
    def from_config(config):
        """Instantiate from configuration dictionary"""
        keys = ['element_list',
                'degree',
                'r_min_map',
                'r_max_map',
                'resolution_map']
        config = {k: v for k, v in config.items() if k in keys}
        return ChemicalSystem(**config)

    def get_composition_tuple(self, geometry):
        """
        Args:
            geometry (ase.Atoms)

        Returns:
            composition_vector (np.ndarray): vector of frequency of
                each element in self.element_list.
        """
        composition_vector = np.zeros(len(self.element_list))
        numbers = geometry.get_atomic_numbers()
        for i, element in enumerate(self.numbers):
            composition_vector[i] = np.sum(numbers == element)
        return composition_vector
