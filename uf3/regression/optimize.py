"""
This module provides functions for optimizing the cutoffs of 2-body and 3-body
terms. If a feature file exist with a higher cutoff, features with lower
cutoffs can be constructed by droping appropriate columns. The functions
provided only work when used with uniform knot spacing.
"""

import numpy as np
from uf3.representation import bspline


def get_bspline_config(
    chemical_system,
    rmin_2b: float,
    rmin_3b: float,
    rmax_2b: float,
    rmax_3b: float,
    knot_spacing_2b: float,
    knot_spacing_3b: float,
    leading_trim: int,
    trailing_trim: int,
):
    """
    Function for getting bspline_config object. We recommend using this function
    to get bspling_config object for-
        1. creating the HDF5 file with larger cutoffs
        2. fitting model with lower cutoffs
    The rmax_2b and rmax_3b values in the returned bspline_config object can be
    slightly different depending on the knot_spacing

    Args:
        chemical_system: data.composition.ChemicalSystem
        rmin_2b (float): Min of 2-body interaction
        rmin_3b (float): Min of 3-body interaction
        rmax_2b (float): Two-body cutoff
        rmax_3b (float): Three-body cutoff. If the 3-body interaction is A-B-C,
            then rmax_3b is the distance between A-B (or equivalently A-C) i.e
            the first or second term in the 3-body rmax of bspline_config.
        knot_spacing_2b (float): knot_spacing for 2-body interaction to create
            the HDF5 file
        knot_spacing_3b (float): knot_spacing for 3-body interaction to create
            the HDF5 file
        leading_trim (int): number of basis functions at leading edge
            to suppress. Useful for ensuring smooth cutoffs.
        trailing_trim (int): number of basis functions at trailing edge
            to suppress. Useful for ensuring smooth cutoffs.

    Returns:
        bspline_config: bspline.BSplineBasis
    """
    if not (
        np.isclose((rmax_2b - rmin_2b) % knot_spacing_2b, knot_spacing_2b)
        or np.isclose((rmax_2b - rmin_2b) % knot_spacing_2b, 0)
    ):

        raise ValueError("Provided rmax_2b does not conatin integer number of\n\
                knots, seperated by knot_spacing_2b")

    if not (
        np.isclose((rmax_3b - rmin_3b) % knot_spacing_3b, knot_spacing_3b)
        or np.isclose((rmax_3b - rmin_3b) % knot_spacing_3b, 0)
    ):

        raise ValueError("Provided rmax_3b does not conatin integer number of\n\
                knots, seperated by knot_spacing_3b")

    if leading_trim != 0:
        raise ValueError("Currrent version is only tested for leading_trim=0")

    if trailing_trim != 3:
        raise ValueError("Currrent version is only tested for trailing_trim=3")

    rmax_3b_double = rmax_3b * 2
    if not (
        np.isclose(((rmax_3b_double - rmin_3b) % knot_spacing_3b), 0)
        or np.isclose(((rmax_3b_double - rmin_3b) % knot_spacing_3b), knot_spacing_3b)
    ):
        raise ValueError(
            "Provided (rmax_3b-rmin_3b) contains integer number of knots \n\
                sperated by knot_spacing_3b, but rmax_3b_double does not. \n\
                Consider changing rmin_3b, rmax_3b, knot_spacing_3b so that \n\
                the following conditions are satisfied- \n\
                --(rmax_3b - rmin_3b)/knot_spacing_3b == integer \n\
                --(rmax_3b_double - rmin_3b)//knot_spacing_3b == integer, \n\
                    where rmax_3b_double = 2*rmax_3b, calculated internally"
        )

    reso_2b = round((rmax_2b - rmin_2b) / knot_spacing_2b)
    reso_3b = round((rmax_3b - rmin_3b) / knot_spacing_3b)

    reso_3b_double = round((rmax_3b_double - rmin_3b) / knot_spacing_3b)

    r_min_map = {i: rmin_2b for i in chemical_system.interactions_map[2]}
    r_min_map.update(
        {
            i: [rmin_3b, rmin_3b, rmin_3b]
            for i in chemical_system.interactions_map[3]
        }
    )

    r_max_map = {i: rmax_2b for i in chemical_system.interactions_map[2]}
    r_max_map.update(
        {
            i: [rmax_3b, rmax_3b, rmax_3b_double]
            for i in chemical_system.interactions_map[3]
        }
    )

    resolution_map = {i: reso_2b for i in chemical_system.interactions_map[2]}
    resolution_map.update(
        {
            i: [reso_3b, reso_3b, reso_3b_double]
            for i in chemical_system.interactions_map[3]
        }
    )

    bspline_config = bspline.BSplineBasis(
        chemical_system,
        r_min_map=r_min_map,
        r_max_map=r_max_map,
        resolution_map=resolution_map,
        trailing_trim=trailing_trim,
        leading_trim=leading_trim,
    )

    return bspline_config


def get_lower_cutoffs(original_bspline_config):
    """
    Function for getting cutoff values obtainable by droping columns of HDF5
    file

    Args:
        original_bspline_config: bspline_config used to create the HDF5 feature
            file. This file was produced with larger cutoff.

    Returns:
        Dict: {"lower_rmax_2b":lower_rmax_2b, "lower_rmax_3b":lower_rmax_3b}
            lower_rmax_2b (list): List of possible 2-body cutoffs
            lower_rmax_3b (list): List of possible 3-body cutoffs

    """
    interaction_2b = original_bspline_config.interactions_map[2][0]
    interaction_3b = original_bspline_config.interactions_map[3][0]

    lower_rmax_2b = original_bspline_config.knots_map[interaction_2b][4:-3]
    lower_rmax_3b = original_bspline_config.knots_map[interaction_3b][0][4:-3]

    for i in range(lower_rmax_2b.shape[0]):
        if lower_rmax_2b[i] not in original_bspline_config.knots_map[interaction_2b]:
            raise ValueError("Internal check failed-->2B!!")

    for i in range(lower_rmax_3b.shape[0]):
        if lower_rmax_3b[i] not in original_bspline_config.knots_map[interaction_3b][0]:
            raise ValueError("Internal check failed-->3B_0!!")

    for i in range(lower_rmax_3b.shape[0]):
        if lower_rmax_3b[i] not in original_bspline_config.knots_map[interaction_3b][1]:
            raise ValueError("Internal check failed-->3B_1!!")

    return {"lower_rmax_2b": lower_rmax_2b, "lower_rmax_3b": lower_rmax_3b}


def get_columns_to_drop_2b(
    original_bspline_config, modify_2b_cutoff: float, knot_spacing_2b: float
):
    """
    Function for getting appropriate 2-body feature columns to drop for fitting
    to lower cutoffs

    Args:
        original_bspline_config: bspline_config used to create the HDF5 feature
            file. This file was produced with larger cutoff.
        modify_2b_cutoff (float): Intended 2-body cutoff
        knot_spacing_2b (float): knot_spacing_2b used to create the HDF5 file
    Returns:
        columns_to_drop_2b (list): Should be passed to drop_columns argument of
            fit_from_file
    """
    if original_bspline_config.leading_trim[2] != 0:
        raise ValueError("Currrent version is only tested for leading_trim=0")

    if original_bspline_config.trailing_trim[2] != 3:
        raise ValueError("Currrent version is only tested for trailing_trim=3")

    column_names = original_bspline_config.get_column_names()
    interaction_partitions_num = original_bspline_config.get_interaction_partitions()[0]
    interaction_partitions_posn = original_bspline_config.get_interaction_partitions()[1]

    columns_to_drop_2b = []

    for interaction in original_bspline_config.interactions_map[2]:
        if modify_2b_cutoff not in original_bspline_config.knots_map[interaction]:
            raise ValueError(
                "Provided modify_2b_cutoff is not a knot in the %s interaction"
                % (str(interaction))
            )

        num_columns_to_drop_2b = round(
            (original_bspline_config.knots_map[interaction][-4] - modify_2b_cutoff)
            / knot_spacing_2b
        )

        start_ind_2b = 1 + interaction_partitions_posn[interaction]
        end_ind_2b = start_ind_2b + interaction_partitions_num[interaction]
        columns_to_drop_2b.extend(
            column_names[end_ind_2b - num_columns_to_drop_2b - 3:end_ind_2b - 3]
        )
    return columns_to_drop_2b


def get_columns_to_drop_3b(
    original_bspline_config, modify_3b_cutoff: float, knot_spacing_3b: float
):
    """
    Function for getting appropriate 3-body feature columns to drop for fitting
    to lower cutoffs

    Args:
        original_bspline_config: bspline.BSplineBasis used to create the HDF5
            feature file. This file was produced with larger cutoff.
        modify_3b_cutoff (float): Intended 3-body cutoff. If the 3-body interaction
            is A-B-C, then modify_3b_cutoff is the maximum distance between A-B
            (or A-C) i.e the first or second term in the 3-body rmax of
            bspline_config.
        knot_spacing_3b (float): knot_spacing_3b used to create the HDF5 file
    Returns:
        columns_to_drop_3b (list): Should be passed to drop_columns argument of
            fit_from_file
    """
    if original_bspline_config.leading_trim[3] != 0:
        raise ValueError("Currrent version is only tested for leading_trim=0")

    if original_bspline_config.trailing_trim[3] != 3:
        raise ValueError("Currrent version is only tested for trailing_trim=3")
    column_names = original_bspline_config.get_column_names()
    interaction_partitions_num = original_bspline_config.get_interaction_partitions()[0]
    interaction_partitions_posn = original_bspline_config.get_interaction_partitions()[
        1
    ]

    columns_to_drop_3b = []
    for interaction in original_bspline_config.interactions_map[3]:
        if modify_3b_cutoff not in original_bspline_config.knots_map[interaction][0]:
            raise ValueError(
                "Provided modify_3b_cutoff is not a knot in %s leg of %s interaction"
                % (str((interaction[0], interaction[1])), str(interaction))
            )

        if modify_3b_cutoff not in original_bspline_config.knots_map[interaction][1]:
            raise ValueError(
                "Provided modify_3b_cutoff is not a knot in %s leg of %s interaction"
                % (str((interaction[0], interaction[2])), str(interaction))
            )

        n_drop_3b = round(
            (original_bspline_config.knots_map[interaction][0][-4] - modify_3b_cutoff)
            / knot_spacing_3b
        )
        n_drop_3b_double = int(n_drop_3b * 2)
        start_ind_3b = 1 + interaction_partitions_posn[interaction]
        end_ind_3b = start_ind_3b + interaction_partitions_num[interaction]

        l_space, m_space, n_space = original_bspline_config.knots_map[interaction]
        L = len(l_space) - 4
        M = len(m_space) - 4
        N = len(n_space) - 4
        grid_c = np.chararray((L, M, N))
        grid_c[:] = "0"
        grid_c = np.char.multiply(grid_c, 16)
        grid_c.flat[original_bspline_config.template_mask[interaction]] = \
            column_names[start_ind_3b:end_ind_3b]

        grid_c = np.delete(grid_c, np.s_[grid_c.shape[2] - 3
                        - n_drop_3b_double:grid_c.shape[2] - 3],
                        axis=2)

        grid_c = np.delete(grid_c, np.s_[grid_c.shape[1] - 3
                    - n_drop_3b:grid_c.shape[1] - 3], axis=1)

        grid_c = np.delete(grid_c, np.s_[grid_c.shape[0] - 3
                    - n_drop_3b:grid_c.shape[0] - 3], axis=0)

        columns_to_keep = grid_c[grid_c[:] != b"0000000000000000"].astype("<U16")

        column_to_drop = np.setdiff1d(
            column_names[start_ind_3b:end_ind_3b], columns_to_keep
        )

        columns_to_drop_3b.extend(column_to_drop)
    return columns_to_drop_3b
