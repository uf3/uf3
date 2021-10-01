from typing import Dict, Tuple, Any
import warnings
import ase
from ase.calculators import calculator as ase_calc
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
try:
    import phonopy
    import seekpath
    from phonopy.structure import atoms as phonopy_atoms
    USE_PHONOPY = True
except ImportError:
    USE_PHONOPY = False


def replace_list(text_string: str):
    """Quick formatting of Gamma point"""
    substitutions = {'GAMMA': u'$\\Gamma$'}
    for item in substitutions.items():
        text_string = text_string.replace(item[0], item[1])
    return text_string


def compute_phonon_data(geom: ase.Atoms,
                        calc: ase_calc.Calculator,
                        n_super: int = 5,
                        disp: float = 0.05,
                        resolution: int = 30
                        ) -> Tuple[Any, Dict, Dict]:
    """
    Args:
        geom (ase.Atoms)
        calc (ase_calc.Calculator)
        n_super (int): size of supercell i.e. n_super x n_super x n_super.
        disp (float): displacement in angstroms.
        resolution (int): resolution of phonon spectra to pass to Phonopy.

    Returns:
        force_constants (np.ndarray)
        path_data (dict)
        bands_dict (dict)
    """
    if not USE_PHONOPY:
        warnings.warn("Phonopy could not be imported.", RuntimeWarning)
        return None, dict(), dict()
    # generate supercells with displacements
    pbc = geom.pbc
    scaled_positions = geom.get_scaled_positions()
    unitcell = phonopy_atoms.PhonopyAtoms(symbols=geom.get_chemical_symbols(),
                                          cell=geom.get_cell().array,
                                          scaled_positions=scaled_positions)
    phonon = phonopy.Phonopy(unitcell=unitcell,
                             supercell_matrix=[[n_super, 0, 0],
                                               [0, n_super, 0],
                                               [0, 0, n_super]])
    phonon.generate_displacements(distance=disp)
    supercells = phonon.get_supercells_with_displacements()
    # get force constants
    force_set = []
    for snapshot in supercells:
        positions = snapshot.get_scaled_positions()
        cell = snapshot.get_cell()
        symbols = snapshot.get_chemical_symbols()
        offset = np.sum(cell, axis=0) / 2
        positions = positions + offset
        snapshot = ase.Atoms(symbols=symbols,
                             cell=cell,
                             scaled_positions=positions,
                             pbc=pbc)
        snapshot.calc = calc
        forces = snapshot.get_forces()
        force_set.append(forces)
    phonon.set_forces(force_set)
    phonon.produce_force_constants()
    force_constants = phonon.get_force_constants()
    # get path data with seekpath
    supercell = phonon.get_supercell()
    cell = supercell.get_cell()
    positions = supercell.get_scaled_positions()
    numbers = np.unique(supercell.get_chemical_symbols(),
                        return_inverse=True)[1]
    path_data = seekpath.get_path((cell, positions, numbers))
    # identify bands
    labels = path_data['point_coords']
    band_ranges = []
    for set in path_data['path']:
        band_ranges.append([labels[set[0]], labels[set[1]]])
    bands =[]
    for q_start, q_end in band_ranges:
        band = []
        for i in range(resolution + 1):
            band.append(np.array(q_start)
                        + (np.array(q_end)
                           - np.array(q_start))
                        / resolution * i)
        bands.append(band)
    # get band structure
    phonon.run_band_structure(bands,
                              is_band_connection=False,
                              with_eigenvectors=True)
    bands_dict = phonon.get_band_structure_dict()
    return force_constants, path_data, bands_dict


def plot_phonon_spectrum(path_data: Dict[str, np.ndarray],
                         bands_dict: Dict[str, np.ndarray],
                         ax: plt.Axes = None,
                         **kwargs,
                         ) -> Tuple[plt.Figure, plt.Axes]:
    """
    Args:
        path_data (dict)
        bands_dict (dict)
        ax (plt.axes)
        **kwargs: keyword arguments to pass to plt.plot() (lines)

    Returns:
        fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=300)
    else:
        fig = ax.get_figure()
    distances = bands_dict['distances']
    frequencies = bands_dict['frequencies']
    labels = path_data['path']
    # set colors
    n_colors = len(distances)
    cmap = cm.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, n_colors))
    if 'cmap' in kwargs:
        cmap = cm.get_cmap(kwargs['cmap'])
        colors = cmap(np.linspace(0, 1, n_colors))
        del kwargs['cmap']
    if 'color' in kwargs:
        colors = [kwargs['color']] * n_colors
        del kwargs['color']
    # plot curve segments
    factor = distances[-1][-1]
    for i, freq in enumerate(distances):
        ax.plot(distances[i]/factor,
                frequencies[i],
                color=colors[i],
                **kwargs)
    # high-symmetry points
    x_labels = []
    s_labels = []
    factor = distances[-1][-1]
    for i, freq in enumerate(distances):
        if labels[i][0] == labels[i - 1][1]:
            s_labels.append(replace_list(labels[i][0]))
        else:
            s_labels.append(replace_list(labels[i - 1][1])
                            + '/' + replace_list(labels[i][0]))
        x_labels.append(distances[i][0] / factor)
    s_labels.append(labels[-1][-1])
    x_labels.append(distances[-1][-1] / factor)
    # dividers along path
    ax.vlines(x_labels, 0, 7, linestyle='--', linewidth=1, color='gray',
              alpha=0.5, zorder=-5)
    # axis labels
    ax.set_xticks(x_labels)
    ax.set_xticklabels(s_labels)
    ax.set_ylabel('Frequency [THz]')
    ax.set_xlabel('Wave Vector')
    return fig, ax
