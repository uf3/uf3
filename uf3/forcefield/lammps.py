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
from ase.calculators import lammpsrun as ase_lammpsrun


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
    interaction = "-".join(interaction)
    x_table = knot_sequence[3:-3]
    r_min = knot_sequence[0]
    r_max = knot_sequence[-1]

    n_line = "N {{0}} RSQ {{1:.{0}f}} {{2:.{0}f}}\n".format(rounding)
    p_line = "{{0}} {{1:.{0}f}} {{2:.{0}f}} {{3:.{0}f}}    #".format(rounding)

    lines = [
        "# DATE: {}  UNITS: metal  CONTRIBUTOR: {}".format(date, contributor),
        "# Ultra-Fast Force Field for {}\n".format(interaction),
        "UF_{}".format(interaction),
        n_line.format(len(x_table), r_min, r_max)]

    for i, r in enumerate(x_table):
        bspline_func = interpolate.BSpline(knot_sequence, coefficients, 3)
        e = bspline_func(r)
        f = -bspline_func(r, nu=1)
        line = p_line.format(i + 1, r, e, f)
        lines.append(line)

    line_indices = np.arange(len(x_table)) + 4
    line_indices = np.insert(np.append(line_indices,
                                       line_indices[-1]),
                             0, line_indices[0])
    for i, coefficient in enumerate(coefficients):
        line_idx = line_indices[i]
        lines[line_idx] += " c{0}={1:.10f}".format(i, coefficient)
    text = '\n'.join(lines)
    if filename is not None:
        with open(filename, 'w') as f:
            f.write(text)
    return text
