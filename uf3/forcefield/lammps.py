import warnings
from datetime import datetime
from multiprocessing import pool

import numpy as np
import pandas as pd
from scipy import interpolate
from ase import data as ase_data
from ase.calculators import lammpsrun

from uf3.util import parallel


def export_tabulated_potential(knots,
                               coefficients,
                               interaction,
                               filename=None,
                               contributor=None):
    """
    Args:
        knots (np.ndarray): knot sequence.
        coefficients (np.ndarray): spline coefficients corresponding to knots.
        interaction (tuple): tuple of elements involved e.g. ("A", "B").
        filename (str, None)
        contributor (str, None)
    """
    now = datetime.now()  # current date and time
    date = now.strftime("%m/%d/%Y")
    contributor = contributor or ""
    if not isinstance(interaction[0], str):
        interaction = [ase_data.chemical_symbols[int(z)]
                       for z in interaction]
    interaction = "-".join(interaction)
    x_table = knots[3:-3]
    r_min = knots[0]
    r_max = knots[-1]

    lines = [
        "# DATE: {}  UNITS: metal  CONTRIBUTOR: {}".format(date, contributor),
        "# Ultra-Fast Force Field for {}\n".format(interaction),
        "UF_{}".format(interaction),
        "N {0} RSQ {1:.10f} {2:.10f}\n".format(len(x_table), r_min, r_max)]

    for i, r in enumerate(x_table):
        bspline = interpolate.BSpline(knots, coefficients, 3)
        e = bspline(r)
        f = -bspline(r, nu=1)
        line = "{0} {1:.10f} {2:.10f} {3:.10f}".format(i + 1, r, e, f)
        lines.append(line)

    text = '\n'.join(lines)
    if filename is not None:
        with open(filename, 'w') as f:
            f.write(text)
    return text
