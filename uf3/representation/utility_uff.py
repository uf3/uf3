# Contains helpful function for Ultra Fast Featurization
# More documentation will be added

import numpy as np
import pandas as pd
from uf3.data.geometry import get_supercell_factors
import ase
from ase import symbols as ase_symbols
import h5py

def get_interactions_map_ff(interactions_map):
    interactions_map_ff = []

    for n1body in interactions_map[1]:
        interactions_map_ff.append(ase_symbols.atomic_numbers[n1body])

    for n2body in interactions_map[2]:
        interactions_map_ff.append((ase_symbols.atomic_numbers[n2body[0]],
                                    ase_symbols.atomic_numbers[n2body[1]]))

    if 3 in interactions_map.keys():
        for n3body in interactions_map[3]:
            interactions_map_ff.append((ase_symbols.atomic_numbers[n3body[0]],
                                        ase_symbols.atomic_numbers[n3body[1]],
                                        ase_symbols.atomic_numbers[n3body[2]]))

    return tuple(interactions_map_ff)

def get_2b_knots_map_ff(bspline_config):
    max_num_knots = 0
    for i in bspline_config.interactions_map[2]:
        max_num_knots = max(len(bspline_config.knots_map[i]),
                            max_num_knots)

    knots_map_ff = np.zeros((len(bspline_config.interactions_map[2]),max_num_knots))


    for i, pair in enumerate(bspline_config.interactions_map[2]):
        shape0 = bspline_config.knots_map[pair].shape[0]
        knots_map_ff[i][0:shape0] = bspline_config.knots_map[pair]

    return knots_map_ff

def get_2b_num_knots_ff(bspline_config):
    knots_size_2b_ff = np.zeros(len(bspline_config.interactions_map[2]),dtype=np.int32)

    for i, pair in enumerate(bspline_config.interactions_map[2]):
        knots_size_2b_ff[i] = bspline_config.knots_map[pair].shape[0]

    return knots_size_2b_ff

def get_3b_knots_map_ff(bspline_config):
    max_num_knots = 0
    for i in bspline_config.interactions_map[3]:
        for j in bspline_config.knots_map[i]:
            max_num_knots = max(len(j),max_num_knots)

    knots_map_ff = np.zeros((len(bspline_config.interactions_map[3]),3,
                             max_num_knots))

    for i, trio in enumerate(bspline_config.interactions_map[3]):
        for j, knots in enumerate(bspline_config.knots_map[trio]):
            shape = knots.shape[0]
            knots_map_ff[i][j][0:shape] = knots

    return knots_map_ff

def get_3b_num_knots_ff(bspline_config):
    knots_size_3b_ff = np.zeros((len(bspline_config.interactions_map[3]),3),
                                dtype=np.int32)

    for i, trio in enumerate(bspline_config.interactions_map[3]):
        for j, knots in enumerate(bspline_config.knots_map[trio]):
            knots_size_3b_ff[i][j] = len(knots)

    return knots_size_3b_ff

def get_3b_feature_size(bspline_config):
    nelements = len(bspline_config.chemical_system.element_list)
    n_pairs = int(nelements*(nelements+1)/2)
    return np.array(bspline_config.partition_sizes[nelements+n_pairs:],dtype=np.int32)

def convert_ase_atom_to_array(ase_atom):
    #cell = np.concatenate([[[0],[0],[0]], ase_atom.cell.array],axis=1)
    shape_0 = ase_atom.get_atomic_numbers().shape[0]
    species_pos = np.concatenate([ase_atom.get_atomic_numbers().reshape((shape_0,1)),
                                  ase_atom.positions], axis=1)
    return species_pos

def get_atoms_array(df):
    ase_atom = df['geometry']
    return convert_ase_atom_to_array(ase_atom)

def get_crystal_index(df):
    return np.full(df['geometry'].get_global_number_of_atoms(), fill_value=df['crystal_index'])

def get_cells(df):
    return df['geometry'].cell.array

def get_geom_array_posn(df):
    return df['geometry_array'].shape(0)

def get_scell_factors(df, bspline_config):
    cell = df['geometry'].cell
    r_cut = bspline_config.r_cut
    return get_supercell_factors(cell, r_cut).astype(np.int32)


def get_supercell_array(df, bspline_config):
    ase_atom = df['geometry']
    supercell = get_supercell(ase_atom, bspline_config.r_cut)
    return convert_ase_atom_to_array(supercell)

def get_force_array(df):
    fx = df['fx']
    fy = df['fy']
    fz = df['fz']
    return np.stack([fx,fy,fz],axis=1)

def get_data_for_UltraFastFeaturization(bspline_config, df):
    chemical_system = bspline_config.chemical_system
    interactions_map_ff = get_interactions_map_ff(chemical_system.interactions_map)
    n2b_knots_map_ff = get_2b_knots_map_ff(bspline_config)
    n2b_num_knots_ff = get_2b_num_knots_ff(bspline_config)
    if bspline_config.degree ==3:
        n3b_knots_map_ff = get_3b_knots_map_ff(bspline_config)
        n3b_num_knots_ff = get_3b_num_knots_ff(bspline_config)
        symm_array = np.array(list(bspline_config.symmetry.values()),dtype=np.int32)
        feature_size_3b = get_3b_feature_size(bspline_config)
    else:
        n3b_knots_map_ff = np.zeros(1,dtype=np.float64)
        n3b_num_knots_ff = np.zeros(1,dtype=np.int32)
        symm_array = np.zeros(1,dtype=np.int32)
        feature_size_3b = np.zeros(1,dtype=np.int32)

    df['atoms_array'] = df.apply(get_atoms_array,axis=1)
    atoms_array = np.concatenate(df['atoms_array'].values)

    energy_array = np.array(df["energy"].values, dtype=np.float64)

    df["force_array"] = df.apply(get_force_array, axis=1)
    forces_array = np.concatenate(df["force_array"].values)

    df['crystal_index'] = range(0,len(df))
    df['crystal_index'] = df.apply(get_crystal_index,axis=1)
    crystal_index = np.concatenate(df['crystal_index'].values,axis=0).astype(np.int32)


    df['cell_array'] = df.apply(get_cells,axis=1)
    cell_array = np.stack(df['cell_array'])


    df['supercell_factors'] = df.apply(get_scell_factors,
                                                         axis=1,
                                                         args=[bspline_config])
    supercell_factors = np.stack(df['supercell_factors'])

    geom_array_posn = np.concatenate(df.apply(lambda x: [x['atoms_array'].shape[0]],
                                                     axis=1))
    geom_array_posn = np.cumsum(geom_array_posn)
    geom_array_posn = np.concatenate([[0], geom_array_posn]).astype(np.int32)

    struct_names = [str(i) for i in df.index]

    return [interactions_map_ff, n2b_knots_map_ff, n2b_num_knots_ff,
            n3b_knots_map_ff, n3b_num_knots_ff, symm_array, feature_size_3b,
            atoms_array, energy_array, forces_array, cell_array, 
            crystal_index, supercell_factors, geom_array_posn,
            struct_names]


def open_uff_feature(filename,key):
    h5py_fp = h5py.File(filename)
    if key not in h5py_fp.keys():
        h5py_fp.close()
        raise KeyError('%s not found in %s'%(key,filename))
    
    group = h5py_fp[key]
    if len(group.keys()) != 7:
        raise ValueError('Group %s is not of the right size'%(key))

    df = {}
    column_names = [i.decode('utf-8') for i in group['axis0'][:]]
    struct_names = [i.decode('utf-8') for i in group['axis1_level0'][:]]
    CI, desc_size = np.unique(group['axis1_label0'][:], return_counts=True)
    desc_row_name = [i.decode('utf-8') for i in group['axis1_level1'][:]]

    index = pd.MultiIndex.from_tuples([
        (struct_names[j],desc_row_name[k]) for j in range(0,len(struct_names)) \
                for k in range(0,desc_size[j])])

    data = group['block0_values'][:]

    df = pd.DataFrame(data, index=index, columns=column_names)
    h5py_fp.close()
    return df
