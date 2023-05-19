"""
This module provides functions for optimizing the cutoffs of 2-body and 3-body 
terms. If a feature file exist with a higher cutoff, features with lower cutoffs
can be constructed by droping appropriate columns. The functions provided only 
work when used with uniform knot spacing.
"""

import numpy as np
from uf3.representation import bspline

def get_bspline_config(chemical_system,
                       rmin: float,
                       rmax_2b: float,
                       rmax_3b: float,
                       knot_spacing: float):
    """
    Function for getting bspline_config object. We recommend using this function
    to get bspling_config object for- 
        1. creating the HDF5 file with larger cutoffs
        2. fitting model with lower cutoffs
    The rmax_2b and rmax_3b values in the returned bspline_config object can be 
    slightly different depending on the knot_spacing
    
    Args:
        chemical_system: chemical_system
        rmax_2b (float): Two-body cutoff
        rmax_3b (float): Three-body cutoff. If the 3-body interaction is A-B-C, then
            modify_3b_cutoff is the maximum distance between B and C i.e the third 
            term in the 3-body rmax of bspline_config.
        knot_spacing (float): knot_spacing used to create the HDF5 file

    Returns:
        bspline_config: 
    """
    if rmin!=0:
        raise ValueError("Currrent version is only tested for rmin=0")

    if (rmax_2b-rmin)%knot_spacing!=0:
        if not (np.isclose((rmax_2b-rmin)%knot_spacing,knot_spacing) or \
                np.isclose((rmax_2b-rmin)%knot_spacing,0)):
            raise ValueError("Provided rmax_2b does not conatin integer number of \n\
                knots, seperated by knot_spacing")

    if (rmax_3b-rmin)%knot_spacing!=0:
        if not (np.isclose((rmax_3b-rmin)%knot_spacing,knot_spacing) or \
                np.isclose((rmax_3b-rmin)%knot_spacing,0)):
            raise ValueError("Provided rmax_3b does not conatin integer number of \n\
                knots, seperated by knot_spacing")

    half_rmax_3b = rmax_3b/2
    if (half_rmax_3b%knot_spacing)!=0:
        raise ValueError("Provided rmax_3b contains integer number of knots sperated \n\
                by knot_spacing, but half_rmax_3b does not. Consider changing rmax_3b \n\
                to "+str(rmax_3b+knot_spacing))

    reso_2b = round((rmax_2b - rmin)/knot_spacing)
    reso_3b = round((rmax_3b - rmin)/knot_spacing)

    half_reso_3b = round(reso_3b/2)
    
    r_min_map = {i:rmin for i in chemical_system.interactions_map[2]}
    r_min_map.update({i:[rmin,rmin,rmin] for i in chemical_system.interactions_map[3]})
    
    r_max_map = {i:rmax_2b for i in chemical_system.interactions_map[2]}
    r_max_map.update({i:[half_rmax_3b,half_rmax_3b,rmax_3b] for i in chemical_system.interactions_map[3]})
    
    trailing_trim = 3
    leading_trim = 0
           
    resolution_map = {i:reso_2b for i in chemical_system.interactions_map[2]}
    resolution_map.update({i:[half_reso_3b,half_reso_3b,reso_3b] for i in chemical_system.interactions_map[3]})

    bspline_config = bspline.BSplineBasis(chemical_system, r_min_map=r_min_map,
            r_max_map=r_max_map,resolution_map=resolution_map,
            trailing_trim=trailing_trim,leading_trim=leading_trim)

    return bspline_config


def get_possible_lower_cutoffs(original_bspline_config):
    """
    Function for getting cutoff values obtainable by droping columns of HDF5 file

    Args:
        original_bspline_config: bspline_config used to create the HDF5 feature file.
            This file was produced with larger cutoff.

    Returns:
        rmax_2b_poss (list): List of possible 2-body cutoffs
        rmax_3b_poss (list): List of possible 3-body cutoffs

    """
    interaction_2b = original_bspline_config.interactions_map[2][0]
    interaction_3b = original_bspline_config.interactions_map[3][0]

    rmax_2b_poss = original_bspline_config.knots_map[interaction_2b][9:-3]
    rmax_3b_poss = original_bspline_config.knots_map[interaction_3b][2][9:-3][0::2]

    for i in range(rmax_2b_poss.shape[0]):
        if rmax_2b_poss[i] not in  original_bspline_config.knots_map[interaction_2b]:
            raise ValueError("Internal check failed!!")

    for i in range(rmax_3b_poss.shape[0]):
        if rmax_3b_poss[i] not in original_bspline_config.knots_map[interaction_3b][2]:
            raise ValueError("Internal check failed!!")

    return rmax_2b_poss, rmax_3b_poss



def get_columns_to_drop_2b(original_bspline_config,
                           modify_2b_cutoff: float,
                           knot_spacing: float):
    """
    Function for getting appropriate 2-body feature columns to drop for fitting 
    to lower cutoffs

    Args:
        original_bspline_config: bspline_config used to create the HDF5 feature file.
            This file was produced with larger cutoff.
        modify_2b_cutoff (float): Intended 2-body cutoff
        knot_spacing (float): knot_spacing used to create the HDF5 file

    Returns:
        columns_to_drop_2b (list): Should be passed to drop_columns argument of
            fit_from_file
    """
    column_names = original_bspline_config.get_column_names()
    interaction_partitions_num = original_bspline_config.get_interaction_partitions()[0]
    interaction_partitions_posn = original_bspline_config.get_interaction_partitions()[1]

    columns_to_drop_2b = [] 
    for interaction in original_bspline_config.interactions_map[2]:
        num_columns_to_drop_2b = round((original_bspline_config.knots_map[interaction][-4]-modify_2b_cutoff)/knot_spacing)

        start_ind_2b = 1+interaction_partitions_posn[interaction]
        end_ind_2b = start_ind_2b+interaction_partitions_num[interaction]
        columns_to_drop_2b.extend(column_names[end_ind_2b-num_columns_to_drop_2b-3:end_ind_2b-3])
    return columns_to_drop_2b

def get_columns_to_drop_3b(original_bspline_config,
                           modify_3b_cutoff: float,
                           knot_spacing: float):
    """
    Function for getting appropriate 2-body feature columns to drop for fitting
    to lower cutoffs

    Args:
        original_bspline_config: bspline_config used to create the HDF5 feature file.
            This file was produced with larger cutoff.
        modify_3b_cutoff (float): Intended 3-body cutoff. If the 3-body interaction 
            is A-B-C, then modify_3b_cutoff is the maximum distance between B and C
            i.e the third term in the 3-body rmax of bspline_config.
        knot_spacing (float): knot_spacing used to create the HDF5 file

    Returns:
        columns_to_drop_3b (list): Should be passed to drop_columns argument of
            fit_from_file
    """
    column_names = original_bspline_config.get_column_names()
    interaction_partitions_num = original_bspline_config.get_interaction_partitions()[0]
    interaction_partitions_posn = original_bspline_config.get_interaction_partitions()[1]

    columns_to_drop_3b = []
    for interaction in original_bspline_config.interactions_map[3]:
        num_columns_to_drop_3b = round((original_bspline_config.knots_map[interaction][2][-4]-modify_3b_cutoff)/knot_spacing)
        modify_3b_cutoff_half = modify_3b_cutoff/2
        num_columns_to_drop_3b_half = int(num_columns_to_drop_3b/2)
        start_ind_3b = 1+interaction_partitions_posn[interaction]
        end_ind_3b = start_ind_3b + interaction_partitions_num[interaction]

        l_space, m_space, n_space = original_bspline_config.knots_map[interaction]
        L = len(l_space) - 4
        M = len(m_space) - 4
        N = len(n_space) - 4
        grid_c = np.chararray((L, M, N))
        grid_c[:] = '0' 
        grid_c = np.char.multiply(grid_c,16)
        grid_c.flat[original_bspline_config.template_mask[interaction]] = column_names[start_ind_3b:end_ind_3b]

        grid_c = np.delete(grid_c,np.s_[grid_c.shape[2]-3-num_columns_to_drop_3b:grid_c.shape[2]-3],axis=2) #4
        grid_c = np.delete(grid_c,np.s_[grid_c.shape[1]-3-num_columns_to_drop_3b_half:grid_c.shape[1]-3],axis=1) #2
        grid_c = np.delete(grid_c,np.s_[grid_c.shape[0]-3-num_columns_to_drop_3b_half:grid_c.shape[0]-3],axis=0) #2

        columns_to_keep = grid_c[grid_c[:] != b'0000000000000000'].astype('<U16')

        columns_to_keep_index_3b = np.where(np.isin(column_names[start_ind_3b:end_ind_3b],columns_to_keep))[0]

        column_to_drop = np.setdiff1d(column_names[start_ind_3b:end_ind_3b],columns_to_keep)
        column_to_drop_index = np.where(np.isin(column_names[start_ind_3b:end_ind_3b],column_to_drop))[0]

        columns_to_drop_3b.extend(column_to_drop)
    return columns_to_drop_3b
