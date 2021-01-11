import yaml
import numpy as np
from ase import Atoms

from linearqmml.utils import tqdm_wrap
from linearqmml.utils import natural_sort

from dscribe.utils.geometry import get_extended_system
from scipy.spatial.distance import pdist

class LammpsParser:
    """
    Parse LAMMPS .xyz and dumps. Supports serialization to yaml.

    Suggest usage:
    1. process_dump()
    2. flatten_entries() to retrieve keys, ase Atoms objects, and energies
    """
    def __init__(self, root, stem, element_symbols, element_aliases=None):
        """
        Args:
            root (str): Path to working directory.
            stem (str): Prefix for dataset filenames.
            element_symbols (list): List of element symbols (str).
            element_aliases (dict): Optional mapping of aliases
                for element symbols. LAMMPS users typically use integers
                as element identifiers in inputs/outputs. Defaults to
                integer: symbol mapping based on alphabetical sorting plus
                symbol: symbol identity mapping.
        """
        self.root = root
        self.stem = stem
        self.entry_keys = None
        self.entries = None

        self.ase_atoms = None
        self.element_symbols = element_symbols
        if element_aliases is None:
            sorted_symbols = natural_sort(self.element_symbols)
            self.element_aliases = {i: k for i, k in enumerate(sorted_symbols)}
            self.element_aliases.update({k: k for k in sorted_symbols})
        else:
            self.element_aliases = element_aliases

    def load_yaml(self, filename=None):
        """Load entries from yaml. Defaults to [root]/[stem]_entries.yaml"""
        if filename is None:
            filename = '{}/{}_entries.yaml'.format(self.root, self.stem)
        entries = yaml.load(open(filename, 'r'), Loader=yaml.Loader)
        if self.entry_keys is None:
            self.entry_keys = natural_sort(entries.keys())
        self.entries = {key: entries[key] for key in self.entry_keys}

    def save_yaml(self, filename=None):
        """Save entries to yaml. Defaults to [root]/[stem]_entries.yaml"""
        if filename is None:
            filename = '{}/{}_entries.yaml'.format(self.root, self.stem)
        yaml.load(self.entries, open(filename, 'w'))

    def flatten_entries(self, local=False, indices=None, key_subset=None):
        """
        Flattens entries into lists e.g. for machine learning.

        Args:
            local (bool): uses list of energies if available instead of sum.
            indices (list): optional subset of integers for slicing the
                flattened list of keys.
            key_subset (list): optional list of keys with which to take subset.

        Returns:
            keys (list): Entry names. Often simply the timesteps from LAMMPS.
            geometries (list): List of ase Atoms objects.
            energies (list): List of energies or lists of local energies.
        """
        keys = self.entry_keys
        if self.ase_atoms is None:
            self.ase_atoms = ase_from_entries(self.entries)
        if local:
            energies = {k: v['energy'] for k, v in self.entries.items()}
        else:
            energies = {k: np.sum(v['energy']) for k, v
                        in self.entries.items()}
        if key_subset is not None:
            indices = [keys.index(key) for key in key_subset]
        elif indices is None:
            indices = range(len(keys))
        keys = [keys[i] for i in indices]
        geometries = [self.ase_atoms[k] for k in keys]
        energies = [energies[k] for k in keys]
        return keys, geometries, energies

    def process_dump(self, filename=None, overwrite=False):
        """
        Process custom LAMMPS dump file. Obtain by LAMMPS commands:
            "compute peratom all pe/atom"
            "compute pe all reduce sum c_perat"
            "dump fix_dump all custom 1 ${root}/${stem}.dump id x y z c_perat"

        Args:
            filename: Optional filename. Defaults to [root]/[stem].dump
        """
        if filename is None:
            filename = '{}/{}.dump'.format(self.root, self.stem)
        with open(filename, 'r') as f:
            text = f.read()
            delimit_string = 'ITEM: TIMESTEP\n'
        timestep_chunks = text.strip(delimit_string).split(delimit_string)

        assert self.entries is None or overwrite is True, 'Existing entries.'
        self.entries = {}
        for timestep_text in tqdm_wrap(timestep_chunks):
            # Lists are preallocated and indexed due accomodate
            # shuffled atom indices in parallel LAMMPS simulations.
            lines = timestep_text.splitlines()
            n_atoms = len(lines[8:])
            entry = dict(lattice=None,
                         species=[None] * n_atoms,
                         coords=[None] * n_atoms,
                         energy=[None] * n_atoms)
            lattice_min = [float(lines[idx].split()[0]) for idx in [4, 5, 6]]
            lattice_max = [float(lines[idx].split()[1]) for idx in [4, 5, 6]]
            entry['lattice'] = np.subtract(lattice_max, lattice_min).tolist()

            for line in lines[8:]:
                ind, el, x, y, z, pe = line.split()
                ind = int(ind) - 1
                element = self.element_aliases[el]
                position = np.subtract([float(x), float(y), float(z)],
                                       lattice_min).tolist()
                entry['species'][ind] = element
                entry['coords'][ind] = position
                entry['energy'][ind] = float(pe)
            timestep = int(lines[0])
            self.entries[timestep] = entry
        self.entry_keys = sorted(self.entries.keys())


def ase_from_entries(entries, pbc=True):
    """
    Args:
        entries (dict): nested dictionary of geometries and energies
            from parsing LAMMPS output(s).
        pbc: boolean or list of 3 booleans, one for each dimension.

    Returns:
        ase_atoms (dict): ase Atoms objects.
    """
    ase_atoms = {}
    for key, entry in entries.items():
        ase_atoms[key] = Atoms(positions=entry['coords'],
                                 symbols=entry['species'],
                                 cell=entry['lattice'],
                                 pbc=pbc)
    return ase_atoms


# def get_distances(ase_atoms, radial_cutoff=6):
#     """
#     Args:
#         ase_atoms: nested dictionary of geometries from ase_from_entries.
#
#     Returns:
#         raw_distances (dict): Flattened lists of observed distances per entry
#     """
#     raw_distances = {}
#     for key, atoms in ase_atoms.items():
#         centers = atoms.get_positions()
#         ext_system, cell_indices = get_extended_system(atoms,
#                                                        radial_cutoff,
#                                                        centers)
#         coordinates = ext_system.get_positions()
#         distance_matrix = pdist(coordinates)
#         distance_matrix = distance_matrix[:, :len(centers)]
#         # slice of original atoms
#         distances = distance_matrix[distance_matrix<radial_cutoff].tolist()
#         raw_distances[key] = distances
#     return raw_distances