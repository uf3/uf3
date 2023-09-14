"""
This module provides the ChemicalSystem class for managing quantities related
to elements, composition, and element-element interactions.
"""

from typing import List, Dict, Collection, Tuple, Any
import itertools
import numpy as np
import ase
from ase import symbols as ase_symbols

reference_X = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96,
    'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
    'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114,
    'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}


class ChemicalSystem:
    """
    Handler class for managing quantities related to elements, composition,
    and element-element interactions.
    """
    degree: int
    element_list: Collection[str]
    numbers: List[int]
    interactions: List[Tuple[str]]
    interactions_map: Dict[int, Collection[Tuple[str]]]
    interaction_hashes: Dict[int, np.ndarray]

    def __init__(self,
                 element_list: Collection[str],
                 degree: int = 2
                 ) -> None:
        """
        Args:
            element_list (list): set of elements in chemical system
                e.g. ['Ne', 'Xe'] or [10, 54]
            degree (int): handle N-body interactions
                e.g. 2 to fit pair potentials.
        """
        self.degree = degree
        self.element_list = sort_interaction_symbols(list(set(element_list)),
                                                     fix_first=False)
        self.numbers = [ase_symbols.symbols2numbers(el).pop()
                        for el in self.element_list]
        self.interactions_map = self.get_interactions_map()
        self.interactions = self.get_interactions_list()
        self.interaction_hashes = self.get_interaction_hashes()

    @staticmethod
    def from_config(config):
        return ChemicalSystem.from_dict(config)

    @staticmethod
    def from_dict(config: Dict[Any, Any]):
        """Instantiate from configuration dictionary"""
        keys = ['element_list',
                'degree']
        config = {k: config[k] for k in keys}
        return ChemicalSystem(**config)

    def as_dict(self):
        dump = dict(element_list=self.element_list,
                    degree=self.degree)
        return dump

    def __repr__(self):
        summary = ["ChemicalSystem:",
                   f"    Elements: {self.element_list}",
                   f"    Degree: {self.degree}",
                   f"    Pairs: {self.interactions_map[2]}",
                   ]
        if self.degree > 2:
            summary.append(f"    Trios: {self.interactions_map[3]}")
        # summary.append("    Hashes:")
        # for n in range(2, self.degree + 1):
        #     element_combinations = self.interactions_map[n]
        #     hash_list = self.interaction_hashes[n]
        #     for k, v in zip(element_combinations, hash_list):
        #         summary.append(" " * 8 + f"{str(k)}: {str(v)}")
        return "\n".join(summary)

    def __str__(self):
        return self.__repr__()

    def get_composition_tuple(self, geometry: ase.Atoms) -> np.ndarray:
        """
        Extract composition vector from ase.Atoms object.

        Args:
            geometry (ase.Atoms)

        Returns:
            composition_vector: vector of frequency of
                each element in self.element_list.
        """
        composition_vector = np.zeros(len(self.element_list), dtype=int)
        numbers = geometry.get_atomic_numbers()
        for i, element in enumerate(self.numbers):
            composition_vector[i] = np.sum(numbers == element)
        return composition_vector

    def get_interactions_map(self) -> Dict[int, Collection[Tuple[str]]]:
        """
        Compute interactions map from combinations of elements with
        replacement.

        Returns:
            interactions_map: tuples of element symbols, grouped by degree
                up to self.degree, e.g. two-body (2) and three-body (3).
        """
        interactions_map = dict()
        interactions_map[1] = self.element_list
        cwr = itertools.combinations_with_replacement(self.element_list, 2)
        cwr = [sort_interaction_symbols(symbols) for symbols in cwr]
        interactions_map[2] = sorted(cwr,
                                     key=lambda c: [reference_X[x] for x in c])
        for d in range(3, self.degree + 1):
            combinations = get_element_combinations(self.element_list, d)
            combinations.sort(key=lambda c: [reference_X[x] for x in c])
            interactions_map[d] = combinations
        return interactions_map

    def get_interactions_list(self) -> List[Tuple[str]]:
        """
        Return flattened list of interactions from interactions map.

        Returns:
            interactions_list: list of tuples of element symbols,
                in order of degree.
                e.g. for unary system, ["W", ("W", "W"), ("W", "W", "W")]
        """
        interactions_list = list(self.element_list)
        for i in range(2, self.degree + 1):
            interactions_list.extend(list(self.interactions_map[i]))
        return interactions_list

    def get_interaction_hashes(self) -> Dict[int, np.ndarray]:
        """
        Compute integer hashes for element-element interactions.

        Returns:
            interaction_hashes: mapping of interaction tuples to integer
                hashes based on element numbers, sorted by electronegativity.
        """
        interaction_hashes = {}
        for n in range(2, self.degree + 1):
            element_combinations = self.interactions_map[n]
            numbers = np.array([ase_symbols.symbols2numbers(el_tuple)
                                for el_tuple in element_combinations])
            numbers[:, 1:] = np.sort(numbers[:, 1:], axis=1)
            hash_list = get_szudzik_hash(numbers)
            interaction_hashes[n] = hash_list
        return interaction_hashes


def interactions_to_numbers(interactions):
    if isinstance(interactions, tuple):
        return tuple(ase_symbols.symbols2numbers(interactions))
    elif isinstance(interactions, list):
        return [interactions_to_numbers(item) for item in interactions]
    elif isinstance(interactions, dict):
        return {k: interactions_to_numbers(v)
                for k, v in interactions.items()}
    elif isinstance(interactions, str):
        return ase_symbols.atomic_numbers[interactions]
    else:
        raise ValueError


def sort_elements(symbols):
    symbols = sorted(symbols, key=lambda el: reference_X[el])
    return symbols


def sort_interaction_map(imap: Dict[Tuple[str], Any]) -> Dict[Tuple[str], Any]:
    """Apply sort_interaction_symbols() to each key in a dictionary."""
    return {sort_interaction_symbols(k): v for k, v in imap.items()}


def sort_interaction_symbols(symbols: Collection[str], fix_first=True) -> Tuple:
    """
    Sort interaction tuple by electronegativities.
    For consistency, many-body interactions (i.e. >2) are sorted while fixing
    the first (center) element."""
    if len(symbols) >= 3 and fix_first:
        center = symbols[0]
        symbols = sort_elements(symbols[1:])
        symbols.insert(0, center)
        return tuple(symbols)
    else:
        return tuple(sort_elements(symbols))


def get_electronegativity_sort(numbers: Collection[int]) -> np.ndarray:
    """Query electronegativities given element number(s)."""
    symbols = np.array(ase_symbols.chemical_symbols)[np.array(numbers)]
    array = np.zeros_like(symbols, dtype=float)
    for idx, symbol in enumerate(symbols.flat):
        array.flat[idx] = reference_X[symbol]
    return array


def get_element_combinations(element_list: Collection[str],
                             n: int = 3
                             ) -> List[Tuple[str]]:
    """
    Find chemical interactions from element list based on
        combinations (choose n). First column corresponds to the "center" atom,
        incorporates permutational symmetry for other positions.

    Args:
        element_list (list): symbols of elements in chemical system
        n (int): degree of interactions.

    Returns:
        combinations (list)
    """
    numbers = ase_symbols.symbols2numbers(element_list)
    combinations = np.meshgrid(*[numbers] * n)
    combinations = np.vstack([grid.flatten() for grid in combinations]).T
    # sort by electronegativity of first element per combination
    combo_X = get_electronegativity_sort(combinations)
    center_sort = np.argsort(combo_X[:, 0])
    combinations = combinations[center_sort]
    combo_X = combo_X[center_sort]
    # reorder each combination by electronegativity and remove duplicates
    neighbor_sort = np.argsort(combo_X[:, 1:], axis=1)
    neighbor_numbers = combinations[:, 1:]
    axis_index = np.arange(neighbor_sort.shape[0])[:, None]
    neighbor_numbers = neighbor_numbers[axis_index, neighbor_sort]
    combinations[:, 1:] = neighbor_numbers
    uniq, index = np.unique(combinations, axis=0, return_index=True)
    combinations = uniq[index.argsort()]
    # convert elemental numbers back to symbols
    combinations = [tuple([ase_symbols.chemical_symbols[number]
                           for number in row])
                   for row in combinations]
    return combinations


def szudzik_pair(pairs: np.ndarray) -> np.ndarray:
    """
    Numpy implementation of a pairing function by Matthew Szudzik

    Args:
        pairs (np.ndarray): n x 2 integer array of pairs.

    Returns:
        hash_list (np.ndarray): n x 1 integer array of hashes.
    """
    xy = np.array(pairs)
    x = xy[..., 0]
    y = xy[..., 1]
    hash_list = np.zeros_like(x)
    mask = (x > y)
    hash_list[mask] = x[mask] ** 2 + y[mask]
    hash_list[~mask] = y[~mask] ** 2 + x[~mask] + y[~mask]
    return hash_list


def szudzik_unpair(hash_list: np.ndarray) -> np.ndarray:
    """
    Numpy implementation of a pairing function by Matthew Szudzik

    Args:
        hash_list (np.ndarray): n x 1 integer array of hashes.

    Returns:
        pairs (np.ndarray): n x 2 integer array of pairs.
    """
    b = np.sqrt(hash_list).astype(int)
    a = hash_list - b ** 2
    mask = (a < b)
    pairs = np.zeros((len(hash_list), 2))
    pairs[mask, 0] = b[mask]
    pairs[mask, 1] = a[mask]
    pairs[~mask, 0] = a[~mask] - b[~mask]
    pairs[~mask, 1] = b[~mask]
    return pairs


def get_szudzik_hash(array: np.ndarray) -> np.ndarray:
    """
    Recursive application of pairing function for d columns.

    Args:
        array (np.ndarray): n x d integer array of pairs.

    Returns:
        hash_list (np.ndarray): n x 1 integer array of hashes.
    """
    n_rows, n_cols = array.shape
    hash_list = array[:, 0]
    for column_idx in range(1, n_cols):
        pairs = np.vstack([hash_list, array[:, column_idx]]).T
        hash_list = szudzik_pair(pairs)
    return hash_list


def unpack_szudzik_hash(hash_list: np.ndarray, n_iter: int) -> np.ndarray:
    """
    Recursive application of pairing function for d columns.

    Args:
        hash_list (np.ndarray):  n x 1 integer array of hashes.
        n_iter (int): number of resulting columns.

    Returns:
        integer array of unhashed values (n_iter columns).
    """
    columns = []
    for i in range(n_iter - 1):
        unpacked = szudzik_unpair(hash_list)
        columns.insert(0, unpacked[:, 1])
        hash_list = unpacked[:, 0]
    columns.insert(0, hash_list)
    return np.vstack(columns).T


def get_pair_hashes(species_set, symbols_set,
                    pair_idx):
    """Convenience function for working with element pairs."""
    i_spec, j_spec = species_set
    i_sym, j_sym = symbols_set
    i_where, j_where = pair_idx
    i_spec = i_spec[i_where]
    j_spec = j_spec[j_where]
    i_eln = [reference_X[x] for x in np.array(i_sym)[i_where]]
    j_eln = [reference_X[x] for x in np.array(j_sym)[j_where]]
    pair_eln = np.vstack([i_eln, j_eln]).T
    idx = list(np.ogrid[[slice(x) for x in pair_eln.shape]])
    idx[1] = pair_eln.argsort(1)
    pair_spec = np.vstack([i_spec, j_spec]).T
    pair_spec = pair_spec[tuple(idx)]
    hashes = get_szudzik_hash(pair_spec)
    return hashes


def hash_gather(values, hashes):
    """Arrange entries by hash, given vectors."""
    value_ref = {}
    hash_set = np.sort(np.unique(hashes))

    for pair in hash_set:
        pair_mask = (hashes == pair)
        pair_dists = values[pair_mask]
        value_ref[int(pair)] = pair_dists
    return value_ref


def symbols_to_hash(symbols):
    numbers = [ase_symbols.atomic_numbers[symbol]
               for symbol in symbols]
    numbers = np.array([numbers])
    return get_szudzik_hash(numbers)[0]


def hash_to_symbols(hash_, n=2):
    pair = unpack_szudzik_hash([hash_], n)[0]
    return tuple([ase_symbols.chemical_symbols[int(idx)]
                  for idx in pair])
