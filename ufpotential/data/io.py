import os
import re
import io
import numpy as np
import pandas as pd
import ase
from ase import io as ase_io
from ase.io import lammpsrun as ase_lammpsrun


class DataCoordinator:
    def __init__(self,
                 atoms_key='geometry',
                 energy_key='energy',
                 size_key='size',
                 overwrite=False):
        self.atoms_key = atoms_key
        self.energy_key = energy_key
        self.size_key = size_key
        self.overwrite = overwrite

        self.data = {}
        self.keys = []

    @staticmethod
    def from_config(config):
        """Instantiate from configuration dictionary"""
        keys = ['atoms_key', 'energy_key', 'size_key', 'overwrite']
        config = {k: v for k, v in config.items() if k in keys}
        return DataCoordinator(**config)

    def consolidate(self, remove_duplicates=True, keep='first'):
        """Wrapper for concat_dataframes"""
        dataframes = [self.data[k] for k in self.keys]
        df = concat_dataframes(dataframes,
                               remove_duplicates=remove_duplicates,
                               keep=keep)
        return df

    def load_dataframe(self, dataframe, prefix=None):
        """Load dataframe manually."""
        for key in [self.atoms_key, self.energy_key, self.size_key]:
            if key not in dataframe.columns:
                raise RuntimeError("Missing \"{}\" column.".format(key))
        name_0 = dataframe.index[0]  # existing prefix takes priority
        if isinstance(name_0, str):
            if '_' in name_0:
                prefix = '_'.join(name_0.split('_')[:-1])
        if prefix is None:  # no prefix provided
            prefix = len(self.data)
            pattern = '{}_{{}}'.format(prefix)
            dataframe = dataframe.rename(pattern.format)
        if prefix in self.data:
            print('Data already exists with prefix "{}".', end=' ')
            if self.overwrite is True:
                print('Overwriting...')
                self.data[prefix] = dataframe
            else:
                print('Skipping...')
                return
        else:
            self.data[prefix] = dataframe
            self.keys.append(prefix)

    def dataframe_from_lists(self,
                             geometries,
                             prefix=None,
                             energies=None,
                             forces=None,
                             load=True,
                             **kwargs):
        """Wrapper for prepare_dataframe_from_lists"""
        if prefix is None:
            prefix = len(self.data)
        df = prepare_dataframe_from_lists(geometries,
                                          prefix,
                                          energies=energies,
                                          forces=forces,
                                          atoms_key=self.atoms_key,
                                          energy_key=self.energy_key,
                                          size_key=self.size_key,
                                          **kwargs)
        if load:
            self.load_dataframe(df, prefix=prefix)
        else:
            return df

    def dataframe_from_trajectory(self,
                                  filename,
                                  prefix=None,
                                  load=True,
                                  **kwargs):
        """Wrapper for parse_trajectory"""
        if prefix is None:
            prefix = len(self.data)
        df = parse_trajectory(filename,
                              prefix=prefix,
                              atoms_key=self.atoms_key,
                              energy_key=self.energy_key,
                              size_key=self.size_key,
                              **kwargs)
        if load:
            self.load_dataframe(df, prefix=prefix)
        else:
            return df

    dataframe_from_xyz = dataframe_from_trajectory
    dataframe_from_vasprun = dataframe_from_trajectory

    def dataframe_from_lammps_run(self,
                                  path,
                                  element_aliases,
                                  prefix=None,
                                  column_subs={"PotEng": "energy"},
                                  log_fname="log.lammps",
                                  dump_fname="dump.lammpstrj",
                                  load=True,
                                  **kwargs):
        """Wrapper for parse_lammps_outputs"""
        if prefix is None:
            prefix = len(self.data)
        df = parse_lammps_outputs(path,
                                  element_aliases,
                                  prefix=prefix,
                                  column_subs=column_subs,
                                  log_fname=log_fname,
                                  dump_fname=dump_fname,
                                  atoms_key=self.atoms_key,
                                  size_key=self.size_key,
                                  **kwargs)
        if load:
            self.load_dataframe(df, prefix=prefix)
        else:
            return df


def concat_dataframes(dataframes,
                      remove_duplicates=True,
                      keep='first'):
    """
    Concatenate list of dataframes with optional removal of duplicate keys.

    Args:
        dataframes (list): list of DataFrames to merge
        remove_duplicates (bool)
        keep (str, bool): 'first', 'last', or False.

    Returns:
        df (pandas.DataFrame)
    """
    df = pd.concat(dataframes)
    duplicate_array = df.index.duplicated(keep=keep)
    if np.any(duplicate_array):
        print('Duplicates keys found:', np.sum(duplicate_array))
        if remove_duplicates:
            print('Removing with keep=', keep)
            df = df[~duplicate_array]
    return df


def prepare_dataframe_from_lists(geometries,
                                 prefix=None,
                                 energies=None,
                                 forces=None,
                                 atoms_key='geometry',
                                 energy_key='energy',
                                 size_key='size',
                                 copy=True):
    """
    Convenience function for arranging data into pandas DataFrame
        with expected column names. Extracts energies and forces from
        provided ase.Atoms objects if unspecified. If specified,
        adds/overwrites energies and/or forces in ase.Atoms objects
        via info and arrays attributes. Length of geometries, energies,
        and forces must match.

    Args:
        geometries (list): list of ase.Atoms configurations.
        prefix (str): prefix for DataFrame index.
            e.g. "bulk" -> [bulk_0, bulk_1, bulk_2, ...]
        energies (list or np.ndarray): vector of energy for each geometry.
        forces (list): list of n x 3 arrays of forces for each geometry.
        atoms_key (str): column name for geometries, default "geometry".
            Modify when parsed geometries are part of a larger pipeline.
        energy_key (str): column name for energies, default "energy".
        size_key (str):  column name for number of atoms per geometry,
            default "size".
        copy (bool): copy geometries, energies and forces before modification.

    Returns:
        df (pandas.DataFrame): standard dataframe with columns
           [atoms_key, energy_key, fx, fy, fz]
    """
    if copy:
        geometries = [geom.copy() for geom in geometries]
    geometries = update_geometries_from_calc(geometries)
    # generate dataframe
    default_columns = [atoms_key, energy_key, 'fx', 'fy', 'fz']
    df = pd.DataFrame(columns=default_columns)
    df[atoms_key] = geometries
    scalar_keys = ()
    array_keys = ()
    if energies is not None:
        if copy:
            energies = np.array(energies)
        df[energy_key] = energies
        scalar_keys = ('energy',)  # add energies to ase.Atoms objects
    if forces is not None:
        if copy:
            forces = [array.copy() for array in forces]
        df['fx'] = [np.array(array)[:, 0] for array in forces]
        df['fy'] = [np.array(array)[:, 1] for array in forces]
        df['fz'] = [np.array(array)[:, 2] for array in forces]
        array_keys = ('fx', 'fy', 'fz')  # add forces to ase.Atoms objects
    # If values are provided, overwrite attributes for consistency.
    update_geometries_from_dataframe(df,
                                     scalar_keys=scalar_keys,
                                     array_keys=array_keys)
    # Otherwise, pull energies and forces from objects.
    scalar_keys = ()
    array_keys = ()
    if energies is None:
        scalar_keys = ('energy',)  # get energies from ase.Atoms objects
    if forces is None:
        array_keys = ('fx', 'fy', 'fz')  # get forces from ase.Atoms objects
    df = update_dataframe_from_geometries(df,
                                          atoms_key=atoms_key,
                                          size_key=size_key,
                                          scalar_keys=scalar_keys,
                                          array_keys=array_keys,
                                          inplace=True)
    if prefix is not None:
        pattern = '{}_{{}}'.format(prefix)
        df = df.rename(pattern.format)
    return df


def parse_trajectory(fname,
                     scalar_keys=(),
                     array_keys=(),
                     prefix=None,
                     atoms_key="geometry",
                     energy_key="energy",
                     size_key='size'):
    """
    Wrapper for ase.io.read, which is compatible with
    many file formats (notably VASP's vasprun.xml and extended xyz).
    If available, force information is written to each ase.Atoms object's
    arrays attribute as separate "fx", "fy", and "fz" entries.

    Args:
        fname (str): filename.
        scalar_keys (list): list of ase.Atoms.info keys to query and
            include as a DataFrame column. e.g. ["config_type"].
        array_keys (list): list of ase.Atoms.arrays keys to query and
            include as a DataFrame column. e.g. ["charge"].
        prefix (str): prefix for DataFrame index.
            e.g. "bulk" -> [bulk_0, bulk_1, bulk_2, ...]
        atoms_key (str): column name for geometries, default "geometry".
            Modify when parsed geometries are part of a larger pipeline.
        energy_key (str): column name for energies, default "energy".
        size_key (str):  column name for number of atoms per geometry,
            default "size".

    Returns:
        df (pandas.DataFrame): standard dataframe with columns
           [atoms_key, energy_key, fx, fy, fz]
    """
    geometries = ase_io.read(fname, index=slice(None, None))
    if not isinstance(geometries, list):
        geometries = [geometries]
    geometries = update_geometries_from_calc(geometries,
                                             energy_key=energy_key)
    # create DataFrame
    default_columns = [atoms_key, energy_key, 'fx', 'fy', 'fz']
    scalar_keys = [p for p in scalar_keys
                   if p not in default_columns]
    array_keys = [p for p in array_keys
                  if p not in default_columns]
    columns = default_columns + scalar_keys + array_keys
    df = pd.DataFrame(columns=columns)
    df[atoms_key] = geometries
    # object-dataframe consistency
    df = update_dataframe_from_geometries(df,
                                          atoms_key=atoms_key,
                                          size_key=size_key,
                                          scalar_keys=scalar_keys,
                                          array_keys=array_keys,
                                          inplace=True)
    if prefix is not None:
        pattern = '{}_{{}}'.format(prefix)
        df = df.rename(pattern.format)
    return df


def parse_lammps_outputs(path,
                         element_aliases,
                         prefix=None,
                         column_subs={"PotEng": "energy"},
                         log_fname="log.lammps",
                         dump_fname="dump.lammpstrj",
                         atoms_key="geometry",
                         size_key='size',
                         log_regex=None):
    """
    Convenience wrapper for parsing both LAMMPS log and dump
    in a run directory.

    Args:
        path (str): path to run directory.
        element_aliases (dict): optional map of LAMMPS atom types to species.
        prefix (str): prefix for DataFrame index.
            e.g. "bulk" -> [bulk_0, bulk_1, bulk_2, ...]
        column_subs (dict): column name substitutions for DataFrame.
            Default {"PotEng": "energy"}.
        log_fname: log filenane, default "log.lammps".
        dump_fname (str): dump filename, default "dump.lammpstrj".
        atoms_key (str): column name for geometries, default "geometry".
            Modify when parsed geometries are part of a larger pipeline.
        size_key (str):  column name for number of atoms per geometry,
            default "size".
        log_regex (str): Regular expression for identifying step information.
            Defaults to '\n(Step[^\n]+\n[^A-Za-z]+)(?:Loop time of)'

    Returns:
        df (pandas.DataFrame): Indexed by timestep, containing
            columns from log (e.g. Temp, PotEng) and column containing
            corresponding ase.Atoms snapshots.
    """
    log_path = os.path.join(path, log_fname)
    dump_path = os.path.join(path, dump_fname)
    # Parse log file, yielding a DataFrame
    df_log = parse_lammps_log(log_path, log_regex=log_regex)
    df = df_log.rename(columns=column_subs)
    df[atoms_key] = pd.Series(dtype=object)
    col_idx = df.columns.get_loc(atoms_key)
    log_timesteps = df['Step'].values
    # Parse dump file, querying only timesteps appearing in the log
    snapshots = parse_lammps_dump(dump_path,
                                  element_aliases,
                                  timesteps=log_timesteps)
    log_idxs = np.arange(len(df))
    intersection_idxs = []
    for timestep, geom in snapshots.items():
        # match log timesteps with snapshot timesteps
        i = np.flatnonzero(log_timesteps == timestep)[0]
        idx = log_idxs[i]
        log_timesteps = np.delete(log_timesteps, i)
        log_idxs = np.delete(log_idxs, i)
        intersection_idxs.append(idx)
    for i, (timestep, geom) in enumerate(snapshots.items()):
        log_idx = intersection_idxs[i]  # index of matching log row
        timestep_info = df.iloc[log_idx].to_dict()  # log row
        df.iat[log_idx, col_idx] = geom
        for key, value in timestep_info.items():
            geom.info[key] = value
    # Add geometries to DataFrame and remove timesteps with no geometry.
    df = df.dropna()
    if prefix is not None:
        pattern = '{}_{{}}'.format(prefix)
        df = df.rename(pattern.format)
    # object-dataframe consistency
    df = update_dataframe_from_geometries(df,
                                          atoms_key=atoms_key,
                                          size_key=size_key,
                                          scalar_keys=['energy'],
                                          array_keys=['fx', 'fy', 'fz'],
                                          inplace=True)
    return df


def update_dataframe_from_geometries(df,
                                     scalar_keys=(),
                                     array_keys=(),
                                     atoms_key='geometry',
                                     size_key='size',
                                     inplace=True):
    """Intermediate function for object-dataframe consistency"""
    if not inplace:
        df = df.copy()
    geometries = df[atoms_key]
    scalar_idxs = []
    array_idxs = []
    for scalar in scalar_keys:
        if scalar not in df.columns:
            df[scalar] = pd.Series(dtype=object)
        scalar_idxs.append(df.columns.get_loc(scalar))
    if size_key not in df.columns:
        df[size_key] = pd.Series(dtype=int)
    size_idx = df.columns.get_loc(size_key)
    for array in array_keys:
        if array not in df.columns:
            df[array] = pd.Series(dtype=object)
        array_idxs.append(df.columns.get_loc(array))
    for idx, geom in enumerate(geometries):
        df.iat[idx, size_idx] = len(geom)
        for scalar, scalar_idx in zip(scalar_keys, scalar_idxs):
            try:
                df.iat[idx, scalar_idx] = geom.info[scalar]
            except KeyError:
                continue
        for array, array_idx in zip(array_keys, array_idxs):
            try:
                df.iat[idx, array_idx] = geom.arrays[array]
            except KeyError:
                continue
    return df


def update_geometries_from_calc(geometries,
                                energy_key='energy',
                                force_key='force'):
    """Query attached calculators for energy and forces."""
    for idx, geom in enumerate(geometries):
        try:
            geom.info[energy_key] = geom.calc.get_potential_energy()
        except (ase.calculators.calculator.PropertyNotImplementedError,
                AttributeError):
            pass  # no energy
        try:
            forces = geom.calc.get_forces()
        except (ase.calculators.calculator.PropertyNotImplementedError,
                AttributeError):
            if force_key in geom.arrays:
                forces = geom.arrays[force_key]
            else:
                continue  # no forces
        try:
            geom.new_array('fx', forces[:, 0])
            geom.new_array('fy', forces[:, 1])
            geom.new_array('fz', forces[:, 2])
        except ValueError:  # shape mismatch
            continue
        except RuntimeError:  # array already exists
            continue
    return geometries


def update_geometries_from_dataframe(df,
                                     scalar_keys=(),
                                     array_keys=(),
                                     atoms_key='geometry',
                                     inplace=True):
    """Intermediate function for object-dataframe consistency"""
    geometries = df[atoms_key]
    if not inplace:
        geometries = [geom.copy() for geom in geometries]
    scalar_idxs = [df.columns.get_loc(scalar) for scalar in scalar_keys]
    array_idxs = [df.columns.get_loc(array) for array in array_keys]
    for idx, geom in enumerate(geometries):
        for scalar, scalar_idx in zip(scalar_keys, scalar_idxs):
            geom.info[scalar] = df.iat[idx, scalar_idx]
        for array, array_idx in zip(array_keys, array_idxs):
            try:
                geom.new_array(array, df.iat[idx, array_idx])
            except ValueError:  # shape mismatch
                continue
            except RuntimeError:  # array already exists
                continue
    return geometries


def df_from_tsv_text(text):
    """Convenience function for converting
        tab-separated values (text) into DataFrame."""
    buffer = io.StringIO(text)  # pandas expects file buffer
    df = pd.read_csv(buffer, delim_whitespace=True)
    df = df.set_index("id").sort_index()
    return df


def atoms_from_df(df,
                  element_key='element',
                  element_aliases=None,
                  info=None,
                  **atom_kwargs):
    """
    Create ase.Atoms from DataFrame. Minimum required columns include:
        x, y, z, [element_key]

    Args:
        df (pandas.DataFrame): DataFrame of interest.
        element_key (str): column name corresponding to species.
        element_aliases (dict): optional map of aliases to species
            e.g. for LAMMPS atom types.
        info (dict): optional dictionary of scalars.
        **atom_kwargs: arguments to pass to ase.Atoms, e.g. cell and pbc.

    Returns:
        atoms (ase.Atoms)
    """
    req_keys = ['x', 'y', 'z', element_key]
    info = info or {}
    element_aliases = element_aliases or {}
    positions = df[['x', 'y', 'z']].to_numpy()
    species = df[element_key]
    species = [element_aliases.get(el, el)
               for el in species]  # substitute aliases
    atoms = ase.Atoms(species, positions=positions, **atom_kwargs)
    # Add extra columns, e.g. fx or per-atom quantities, as array entries.
    extra_keys = list(set(df.columns).difference(req_keys))
    for key in extra_keys:
        atoms.new_array(key, df[key].values)
    atoms.info = info
    return atoms


def parse_lammps_log(fname, log_regex=None):
    """
    Args:
        fname (str): filename of log file.
        log_regex (str): Regular expression for identifying step information.
            Defaults to '\n(Step[^\n]+\n[^A-Za-z]+)(?:Loop time of)'

    Returns:
        df_log (pandas.DataFrame)
    """
    log_regex = log_regex or '\n(Step[^\n]+\n[^A-Za-z]+)(?:Loop time of)'
    log_blocks = []
    with open(fname, 'r') as f:
        text = f.read()
        for text_block in re.compile(log_regex).findall(text):
            buffer = io.StringIO(text_block)
            df = pd.read_csv(buffer, delim_whitespace=True)
            log_blocks.append(df)
    df_log = pd.concat(log_blocks)
    return df_log


def parse_lammps_dump(fname, element_aliases, timesteps=None):
    """
    Read LAMMPS text dump file. Expects the following items in the
    thermo_style: id type x y z

    Other items, such as fx and custom computes,
    are added via ase.Atoms.new_array().

    Compatible with large files because the function reads line-by-line
    and, optionally, saves only specified timesteps.

    Args:
        fname (str): filename of dump file.
        element_aliases (dict): map of LAMMSP type to species.
        timesteps (list, np.ndarray): Optional subset of timesteps to parse.
            Note: function expects timesteps to match dump chronologically.
            This behavior is intended to accommodate LAMMPS runs with
            reset_timestep commands.
    Returns:
        snapshots (pandas.Series): Map of timestep to ase.Atoms, allowing
            repeated entries in case of reset_timestep.
    """
    parse_subset = (timesteps is not None)
    timesteps = np.array(timesteps)

    snapshot_index = []
    snapshot_contents = []

    atom_lines = []
    timestep = None
    cell = None
    pbc = None
    cell_displacement = None
    with open(fname, 'r') as f:
        while True:
            line = f.readline()
            if "ITEM: TIMESTEP" in line or not line:
                if timestep is not None:  # consolidate atom data
                    df = df_from_tsv_text('\n'.join(atom_lines))
                    atoms = atoms_from_df(df,
                                          cell=cell,
                                          pbc=pbc,
                                          celldisp=cell_displacement,
                                          element_key='type',
                                          element_aliases=element_aliases)
                    if not parse_subset:
                        snapshot_index.append(timestep)
                        snapshot_contents.append(atoms)
                    else:
                        if timestep in timesteps:
                            snapshot_index.append(timestep)
                            snapshot_contents.append(atoms)
                            idx = np.flatnonzero(timesteps == timestep)[0]
                            # delete first occurrence of matching timestep
                            timesteps = np.delete(timesteps, idx)
                            if len(timesteps) == 0:
                                # finish early if all requested have been
                                # parsed. May not trigger if a requested
                                # timestep is absent from the dump.
                                break
                timestep = int(f.readline())
                atom_lines = []  # reset timestep data
            elif "ITEM: NUMBER OF ATOMS" in line:
                n_atoms = int(f.readline())  # parsed but not necessary
            elif "ITEM: BOX BOUNDS" in line:  # cell data
                conditions = line.replace("ITEM: BOX BOUNDS ", "").split()
                a_line = f.readline().split()
                b_line = f.readline().split()
                c_line = f.readline().split()
                cell_data = np.array([a_line, b_line, c_line])
                cell_data = cell_data.astype(float)
                cell_bounds = cell_data[:, :2].reshape(6, 1).flatten()
                if len(conditions) < 3:  # nonperiodic
                    pbc = (False, False, False)
                    off_diag = (0.0, 0.0, 0.0)
                elif len(conditions) == 3:  # orthogonal simulation cell
                    pbc = [('p' in condition.lower())
                           for condition in conditions]
                    off_diag = (0.0, 0.0, 0.0)
                else:  # triclinic simulation cell
                    # tilt_factors = conditions[:3]
                    pbc = [('p' in condition.lower())
                           for condition in conditions[3:]]
                    off_diag = cell_data[:, 2]
                c_data = ase_lammpsrun.construct_cell(cell_bounds, off_diag)
                cell, cell_displacement = c_data
            elif "ITEM: ATOMS" in line:  # header
                atom_lines.append(line.replace("ITEM: ATOMS ", ""))
            else:  # atom data
                atom_lines.append(line)
    snapshots = pd.Series(index=snapshot_index,
                          data=snapshot_contents)
    return snapshots
