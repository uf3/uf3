import warnings
from datetime import datetime
from multiprocessing import pool
import numpy as np
import pandas as pd
from scipy import interpolate
from ase import data as ase_data
from uf3.util import parallel
from ase.io import lammpsdata as ase_lammpsdata
from ase.calculators import lammps as ase_lammps
from ase.calculators import lammpslib
from ase.io import lammpsrun


RELAX_LINES = ["fix fix_relax all box/relax iso 0.0 vmax 0.001",
               "min_style cg",
               "minimize 1e-25 1e-25 5000 10000"]


def write_lammps_data(filename, geom, element_list, **kwargs):
    """
    Args:
        filename (str):
        geom (ase.Atoms):
        element_list (list)
    """
    cell = geom.get_cell()
    prism = ase_lammps.Prism(cell)
    ase_lammpsdata.write_lammps_data(filename,
                                     geom,
                                     specorder=element_list,
                                     force_skew=True,
                                     prismobj=prism,
                                     **kwargs)


def export_tabulated_potential(knot_sequence,
                               coefficients,
                               interaction,
                               grid=None,
                               filename=None,
                               contributor=None,
                               rounding=6):
    """
    Args:
        knot_sequence (np.ndarray): knot sequence.
        coefficients (np.ndarray): spline coefficients corresponding to knots.
        interaction (tuple): tuple of elements involved e.g. ("A", "B").
        filename (str, None)
        contributor (str, None)
        rounding (int): number of decimal digits to print.
    """
    now = datetime.now()  # current date and time
    date = now.strftime("%m/%d/%Y")
    contributor = contributor or ""
    if not isinstance(interaction[0], str):
        interaction = [ase_data.chemical_symbols[int(z)]
                       for z in interaction]
    interaction = "-".join(interaction)  # e.g. W-W. Ne-Xe
    # LAMMPS' pair_style table performs interpolation internally.
    if grid is None:  # default: equally-spaced 100 samples
        grid = 100
    if isinstance(grid, int):  # integer number of equally-spaced samples
        x_table = np.linspace(knot_sequence[0], knot_sequence[-1], grid)
    else:  # custom 1D grid
        x_table = grid

    n_line = "N {}\n"
    p_line = "{{0}} {{1:.{0}f}} {{2:.{0}f}} {{3:.{0}f}}".format(rounding)

    lines = [
        "# DATE: {}  UNITS: metal  CONTRIBUTOR: {}".format(date, contributor),
        "# Ultra-Fast Force Field for {}\n".format(interaction),
        "UF_{}".format(interaction),
        n_line.format(len(x_table))]

    for i, r in enumerate(x_table):
        bspline_func = interpolate.BSpline(knot_sequence, coefficients, 3)
        e = bspline_func(r) * 2  # LAMMPS does not double-count bonds
        f = -bspline_func(r, nu=1) * 2  # introduce factor of 2 for consistency
        line = p_line.format(i + 1, r, e, f)
        lines.append(line)
    text = '\n'.join(lines)
    if filename is not None:
        with open(filename, 'w') as f:
            f.write(text)
    return text


class UFLammps(lammpslib.LAMMPSlib):
    """
    Function inspired by ASE.lammpslib.LAMMPSlib.propagate()
    """
    def __init__(self, *args, **kwargs):
        lammpslib.LAMMPSlib.__init__(self, *args, **kwargs)

    def relax(self, atoms):
        """"atoms: Atoms object
            Contains positions, unit-cell, ...
        """
        if not self.started:
            self.start_lammps()
        if not self.initialized:
            self.initialise_lammps(atoms)
        if self.parameters.atom_types is None:
            raise NameError("atom_types are mandatory.")
        self.set_lammps_pos(atoms)
        # Additional commands
        if self.parameters.amendments is not None:
            for cmd in self.parameters.amendments:
                self.lmp.command(cmd)
        # Relax
        for command in RELAX_LINES:
            self.lmp.command(command)
        # read variables that require version-specific handling
        try:
            pos = self.lmp.numpy.extract_atom("x")
            forces = ase_lammps.convert(self.lmp.numpy.extract_atom("f"),
                                        "force", self.units, "ASE")
            nsteps = self.lmp.extract_global('ntimestep')
        except AttributeError:  # older versions of LAMMPS (e.g. April 2020)
            nsteps = self.lmp.extract_global('ntimestep', 0)
            n_atoms = self.lmp.extract_global('natoms', 0)
            pos = np.zeros((n_atoms, 3))
            forces = np.zeros((n_atoms, 3))
            x_read = self.lmp.extract_atom('x', 3)
            f_read = self.lmp.extract_atom('f', 3)
            for i in range(n_atoms):
                for j in range(3):
                    pos[i, j] = x_read[i][j]
                    forces[i, j] = f_read[i][j]
        # Update positions
        pos = ase_lammps.convert(pos, "distance", self.units, "ASE")
        atoms.set_positions(pos)
        # Update cell
        lammps_cell = self.lmp.extract_box()
        boxlo, boxhi, xy, yz, xz, periodicity, box_change = lammps_cell
        celldata = np.array([[boxlo[0], boxhi[0], xy],
                             [boxlo[1], boxhi[1], xz],
                             [boxlo[2], boxhi[2], yz]])
        diagdisp = celldata[:, :2].reshape(6, 1).flatten()
        offdiag = celldata[:, 2]
        cell, celldisp = lammpsrun.construct_cell(diagdisp, offdiag)
        cell = ase_lammps.convert(cell, "distance", self.units, "ASE")
        celldisp = ase_lammps.convert(celldisp, "distance", self.units, "ASE")
        atoms.set_cell(cell)
        atoms.set_celldisp(celldisp)
        # Extract energy
        self.results['energy'] = ase_lammps.convert(
            self.lmp.extract_variable('pe', None, 0),
            "energy", self.units, "ASE"
        )
        self.results['free_energy'] = self.results['energy']
        # Extract stresses
        stress = np.empty(6)
        stress_vars = ['pxx', 'pyy', 'pzz', 'pyz', 'pxz', 'pxy']
        for i, var in enumerate(stress_vars):
            stress[i] = self.lmp.extract_variable(var, None, 0)
        stress_mat = np.zeros((3, 3))
        stress_mat[0, 0] = stress[0]
        stress_mat[1, 1] = stress[1]
        stress_mat[2, 2] = stress[2]
        stress_mat[1, 2] = stress[3]
        stress_mat[2, 1] = stress[3]
        stress_mat[0, 2] = stress[4]
        stress_mat[2, 0] = stress[4]
        stress_mat[0, 1] = stress[5]
        stress_mat[1, 0] = stress[5]
        stress[0] = stress_mat[0, 0]
        stress[1] = stress_mat[1, 1]
        stress[2] = stress_mat[2, 2]
        stress[3] = stress_mat[1, 2]
        stress[4] = stress_mat[0, 2]
        stress[5] = stress_mat[0, 1]
        self.results['stress'] = ase_lammps.convert(
            -stress, "pressure", self.units, "ASE")
        self.results['forces'] = forces.copy()
        self.results['nsteps'] = nsteps
        self.results['volume'] = atoms.get_volume()
        self.atoms = atoms.copy()
        if not self.parameters.keep_alive:
            self.lmp.close()