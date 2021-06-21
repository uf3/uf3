import itertools
from ase import symbols as ase_symbols
import numpy as np

reference_X = dict(X=0, H=2.2, He=0, Li=0.98, Be=1.57, B=2.04, C=2.55, N=3.04,
                   O=3.44, F=3.98, Ne=0, Na=0.93, Mg=1.31, Al=1.61, Si=1.9,
                   P=2.19, S=2.58, Cl=3.16, Ar=0, K=0.82, Ca=1.0, Sc=1.36,
                   Ti=1.54, V=1.63, Cr=1.66, Mn=1.55, Fe=1.83, Co=1.88,
                   Ni=1.91, Cu=1.9, Zn=1.65, Ga=1.81, Ge=2.01, As=2.18,
                   Se=2.55, Br=2.96, Kr=3.0, Rb=0.82, Sr=0.95, Y=1.22, Zr=1.33,
                   Nb=1.6, Mo=2.16, Tc=1.9, Ru=2.2, Rh=2.28, Pd=2.2, Ag=1.93,
                   Cd=1.69, In=1.78, Sn=1.96, Sb=2.05, Te=2.1, I=2.66, Xe=2.6,
                   Cs=0.79, Ba=0.89, La=1.1, Ce=1.12, Pr=1.13, Nd=1.14,
                   Pm=1.13, Sm=1.17, Eu=1.2, Gd=1.2, Tb=1.1, Dy=1.22, Ho=1.23,
                   Er=1.24, Tm=1.25, Yb=1.1, Lu=1.27, Hf=1.3, Ta=1.5, W=2.36,
                   Re=1.9, Os=2.2, Ir=2.2, Pt=2.28, Au=2.54, Hg=2.0, Tl=1.62,
                   Pb=2.33, Bi=2.02, Po=2.0, At=2.2, Rn=2.2, Fr=0.7, Ra=0.9,
                   Ac=1.1, Th=1.3, Pa=1.5, U=1.38, Np=1.36, Pu=1.28, Am=1.3,
                   Cm=1.3, Bk=1.3, Cf=1.3, Es=1.3, Fm=1.3, Md=1.3, No=1.3,
                   Lr=1.3, Rf=0, Db=0, Sg=0, Bh=0, Hs=0, Mt=0, Ds=0, Rg=0,
                   Cn=0, Nh=0, Fl=0, Mc=0, Lv=0, Ts=0, Og=0)


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
        numbers = {el: ase_symbols.symbols2numbers(el).pop()
                   for el in element_list}
        self.element_list = sort_interaction_symbols(element_list)
        self.numbers = [numbers[el] for el in self.element_list]
        self.degree = degree
        self.interactions_map = {}

        cwr = itertools.combinations_with_replacement(self.element_list, 2)
        self.interactions_map[2] = sorted(cwr)
        for d in range(3, degree+1):
            combinations = get_element_combinations(element_list, d)
            self.interactions_map[d] = combinations

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


def sort_interaction_symbols(symbols):
    """
    Sort interaction tuple by electronegativities.
    For consistency, many-body interactions (i.e. >2) are sorted while fixing
    the first (center) element."""
    if len(symbols) > 3:
        center = symbols[0]
        symbols = sorted(symbols[1:], key=lambda el: reference_X[el])
        symbols.insert(0, center)
        return tuple(symbols)
    else:
        return tuple(sorted(symbols, key=lambda el: reference_X[el]))


def get_electronegativity_sort(numbers):
    """Query electronegativities given element number(s)."""
    symbols = np.array(ase_symbols.chemical_symbols)[np.array(numbers)]
    array = np.zeros_like(symbols, dtype=float)
    for idx, symbol in enumerate(symbols.flat):
        array.flat[idx] = reference_X[symbol]
    return array


def get_element_combinations(element_list, n=3):
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