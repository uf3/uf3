import itertools
import ase
import numpy as np


class ChemicalSystem:
    """
    -Manage parameters related to composition
    -e.g. pair combinations of a binary/ternary system
    """
    def __init__(self,
                 element_list,
                 degree=2,
                 r_min_map=None,
                 r_max_map=None,
                 resolution_map=None):
        """
        Args:
            element_list (list): set of elements in chemical system
                e.g. ['Ne', 'Xe'] or [10, 54]
            degree (int): handle N-body interactions
                e.g. 2 to fit pair potentials.
            r_min_map (dict): map of minimum pair distance per interaction.
                If unspecified, defaults to 0.0 for all interactions.
                e.g. {(A-A): 2.0, (A-B): 3.0, (B-B): 4.0}
            r_max_map (dict): map of maximum pair distance per interaction.
                If unspecified, defaults to 6.0 angstroms for all interactions,
                which probably encompasses 2nd-nearest neighbors.
            resolution_map (dict): map of resolution (number of knot intervals)
                per interaction. If unspecified, defaults to 20 for all
                interactions.
        """
        self.element_list = element_list
        self.numbers = ase.symbols.symbols2numbers(element_list)

        self.interactions_map = {}
        for d in range(2, degree+1):
            cwr = itertools.combinations_with_replacement(element_list, d)
            self.interactions_map[d] = sorted(cwr)
        # Minimum pair distance per interaction.
        if r_min_map is None:
            r_min_map = {}
        self.r_min_map = r_min_map
        # Maximum pair distance per interaction.
        if r_max_map is None:
            r_max_map = {}
        self.r_max_map = r_max_map
        # Resolution (knot intervals) per interaction.
        if resolution_map is None:
            resolution_map = {}
        self.resolution_map = resolution_map
        # Default values
        for pair in self.interactions_map[2]:
            self.r_min_map[pair] = self.r_min_map.get(pair, 0.0)
            self.r_max_map[pair] = self.r_max_map.get(pair, 6.0)
            self.resolution_map[pair] = self.resolution_map.get(pair, 20)
        # consistency check
        param_maps = [self.r_min_map, self.r_max_map, self.resolution_map]
        assert all([[el_tuple in param_map for el_tuple
                     in self.interactions_map[degree]]
                    for param_map in param_maps])

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
