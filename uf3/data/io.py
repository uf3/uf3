import os
import re
import io
import fnmatch
from typing import List, Dict, Collection, Tuple, Any, Union
import numpy as np
import pandas as pd
import tables
import ase
from ase import io as ase_io
from ase import db as ase_db
from ase.db import core as db_core
from ase.io import lammpsrun as ase_lammpsrun
from ase.calculators import singlepoint
from ase.calculators import calculator as ase_calc
from uf3.util import subsample


class DataCoordinator:
    """
    -Load data from files, e.g. LAMMPS and VASP outputs
    -Prepare standardized DataFrames for representation
    """
    def __init__(self,
                 atoms_key='geometry',
                 energy_key='energy',
                 force_key='force',
                 size_key='size',
                 overwrite=False
                 ):
        """
        Args:
            atoms_key (str): column name for geometries, default "geometry".
                Modify when parsed geometries are part of a larger pipeline.
            energy_key (str): column name for energies, default "energy".
            force_key (str): identifier for forces, default "force".
            size_key (str):  column name for number of atoms per geometry,
                default "size".
            overwrite (bool): Allow overwriting of existing DataFrame
                with matching key when loading.
        """
        self.atoms_key = atoms_key
        self.energy_key = energy_key
        self.force_key = force_key
        self.size_key = size_key
        self.overwrite = overwrite

        self.data = {}
        self.keys = []

    @staticmethod
    def from_config(config):
        """Instantiate from configuration dictionary"""
        keys = ['atoms_key',
                'energy_key',
                'force_key',
                'size_key',
                'overwrite']
        config = {k: v for k, v in config.items() if k in keys}
        return DataCoordinator(**config)

    def __repr__(self):
        summary = ["DataCoordinator:",
                   ]
        if len(self.keys) == 0:
            summary.append(f"    Datasets: None")
        else:
            summary.append(f"    Datasets: {len(self.keys)} ({self.keys})")
        return "\n".join(summary)

    def __str__(self):
        return self.__repr__()

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
            print('Data already exists with prefix "{}".'.format(prefix),
                  end=' ')
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
                                          force_key=self.force_key,
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
                                  energy_key=None,
                                  force_key=None,
                                  **kwargs):
        """Wrapper for parse_trajectory"""
        if prefix is None:
            prefix = len(self.data)
        if energy_key is None:
            energy_key = self.energy_key
        if force_key is None:
            force_key = self.force_key
        df = parse_trajectory(filename,
                              prefix=prefix,
                              atoms_key=self.atoms_key,
                              energy_key=energy_key,
                              force_key=force_key,
                              size_key=self.size_key,
                              **kwargs)

        if energy_key != self.energy_key:
            df.rename(columns={energy_key: self.energy_key},
                      inplace=True)
        if load:
            self.load_dataframe(df, prefix=prefix)
        else:
            return df

    dataframe_from_xyz = dataframe_from_trajectory
    dataframe_from_vasprun = dataframe_from_trajectory

    def dataframe_from_lammps_run(self,
                                  path,
                                  lammps_aliases,
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
                                  lammps_aliases,
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


def concat_dataframes(dataframes: List[pd.DataFrame],
                      remove_duplicates: bool = True,
                      keep: str = 'first'
                      ) -> pd.DataFrame:
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
            print('Unique keys:', len(df))
    return df


def prepare_dataframe_from_lists(geometries: List[ase.Atoms],
                                 prefix: str = None,
                                 energies: List[float] = None,
                                 forces: List[np.ndarray] = None,
                                 atoms_key: str = 'geometry',
                                 energy_key: str = 'energy',
                                 force_key: str = 'force',
                                 size_key: str = 'size',
                                 copy: bool = True
                                 ) -> pd.DataFrame:
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
        force_key (str): identifier for forces, default "force".
        size_key (str):  column name for number of atoms per geometry,
            default "size".
        copy (bool): copy geometries, energies and forces before modification.

    Returns:
        df (pandas.DataFrame): standard dataframe with columns
           [atoms_key, energy_key, fx, fy, fz]
    """
    if copy:
        geometries = [geom.copy() for geom in geometries]
    geometries = update_geometries_from_calc(geometries,
                                             energy_key=energy_key,
                                             force_key=force_key)
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


def parse_trajectory(fname: str,
                     scalar_keys: List[str] = (),
                     array_keys: List[str] = (),
                     prefix: str = None,
                     atoms_key: str = "geometry",
                     energy_key: str = "energy",
                     force_key: str = 'force',
                     size_key: str = 'size'):
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
        force_key (str): identifier for forces, default "force".
        size_key (str):  column name for number of atoms per geometry,
            default "size".

    Returns:
        df (pandas.DataFrame): standard dataframe with columns
           [atoms_key, energy_key, fx, fy, fz]
    """
    extension = os.path.splitext(fname)[-1]
    kws = ['mysql', 'postgres', 'mariadb']
    if extension in ['.db', '.json'] or any([kw in fname for kw in kws]):
        # handle differently to retrieve attached names instead of reindexing
        geometries = read_database(fname, index=slice(None, None))
        new_index = [geom.info.get('row_name', None) for geom in geometries]
        index_errors = new_index.count(None)
        if index_errors > 1:
            new_index = None
    else:  # flexible read function for a variety of filetypes
        geometries = ase_io.read(fname, index=slice(None, None))
        new_index = None
    if not isinstance(geometries, list):
        geometries = [geometries]
    geometries = update_geometries_from_calc(geometries,
                                             energy_key=energy_key,
                                             force_key=force_key)
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
    scalar_keys = scalar_keys + [energy_key]
    array_keys = array_keys + ["fx", "fy", "fz"]
    df = update_dataframe_from_geometries(df,
                                          atoms_key=atoms_key,
                                          size_key=size_key,
                                          scalar_keys=scalar_keys,
                                          array_keys=array_keys,
                                          inplace=True)
    if new_index is not None:
        df.index = new_index
        print('Loaded index from file:', fname)
    elif prefix is not None:
        pattern = '{}_{{}}'.format(prefix)
        df = df.rename(pattern.format)
    return df


def read_database(filename: str, index: bool = None, **kwargs):
    """
    Args:
        filename (str)
        index(slice): Default (None, None, 1)

    Returns:
        list of ase.Atoms objects from database.
    """
    if index is None:
        index = slice(None, None)
    db = ase.db.connect(filename, serial=True, **kwargs)

    start, stop, _ = index.indices(db.count())
    if start == stop:
        return
    geometries = []
    for row in db.select(offset=start, limit=stop - start):
        geom = row.toatoms(add_additional_information=True)
        key_value_pairs = dict(geom.info['key_value_pairs'])
        del geom.info['key_value_pairs']
        geom.info = {**geom.info, **key_value_pairs}
        geometries.append(geom)
    return geometries


def parse_lammps_outputs(path: str,
                         lammps_aliases: Dict[int, str],
                         prefix: str = None,
                         column_subs: Dict[str, str] = {"PotEng": "energy"},
                         log_fname: str = "log.lammps",
                         dump_fname: str = "dump.lammpstrj",
                         atoms_key: str = "geometry",
                         size_key: str = 'size',
                         log_regex: str = None
                         ) -> pd.DataFrame:
    """
    Convenience wrapper for parsing both LAMMPS log and dump
    in a run directory.

    Args:
        path (str): path to run directory.
        lammps_aliases (dict): optional map of LAMMPS atom types to species.
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
                                  lammps_aliases,
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
    df = df.iloc[intersection_idxs]
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


def update_dataframe_from_geometries(df: pd.DataFrame,
                                     scalar_keys: List[str] = (),
                                     array_keys: List[str] = (),
                                     atoms_key: str = 'geometry',
                                     size_key: str = 'size',
                                     inplace: bool = True
                                     ) -> pd.DataFrame:
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


def update_geometries_from_calc(geometries: List[ase.Atoms],
                                energy_key: str = 'energy',
                                force_key: str = 'force'
                                ) -> List[ase.Atoms]:
    """Query attached calculators for energy and forces."""
    for idx, geom in enumerate(geometries):
        try:
            geom.info[energy_key] = geom.calc.get_potential_energy()
        except (ase_calc.PropertyNotImplementedError,
                AttributeError):
            pass  # no energy
        try:
            forces = geom.calc.get_forces()
        except (ase_calc.PropertyNotImplementedError,
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


def update_geometries_from_dataframe(df: pd.DataFrame,
                                     scalar_keys: List[str] = (),
                                     array_keys: List[str] = (),
                                     atoms_key: str = 'geometry',
                                     inplace: bool = True
                                     ) -> List[ase.Atoms]:
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


def df_from_tsv_text(text: str) -> pd.DataFrame:
    """Convenience function for converting
        tab-separated values (text) into DataFrame."""
    buffer = io.StringIO(text)  # pandas expects file buffer
    df = pd.read_csv(buffer, delim_whitespace=True)
    df = df.set_index("id").sort_index()
    return df


def atoms_from_df(df: pd.DataFrame,
                  element_key: str = 'element',
                  lammps_aliases: Dict[int, str] = None,
                  info: Dict[str, float] = None,
                  **atom_kwargs
                  ) -> ase.Atoms:
    """
    Create ase.Atoms from DataFrame. Minimum required columns include:
        x, y, z, [element_key]

    Args:
        df (pandas.DataFrame): DataFrame of interest.
        element_key (str): column name corresponding to species.
        lammps_aliases (dict): optional map of aliases to species
            e.g. for LAMMPS atom types.
        info (dict): optional dictionary of scalars.
        **atom_kwargs: arguments to pass to ase.Atoms, e.g. cell and pbc.

    Returns:
        atoms (ase.Atoms)
    """
    req_keys = ['x', 'y', 'z', element_key]
    info = info or {}
    lammps_aliases = lammps_aliases or {}
    positions = df[['x', 'y', 'z']].to_numpy()
    species = df[element_key]
    species = [lammps_aliases.get(el, el)
               for el in species]  # substitute aliases
    atoms = ase.Atoms(species, positions=positions, **atom_kwargs)
    # Add extra columns, e.g. fx or per-atom quantities, as array entries.
    extra_keys = list(set(df.columns).difference(req_keys))
    for key in extra_keys:
        atoms.new_array(key, df[key].values)
    atoms.info = info
    return atoms


def parse_lammps_log(fname: str, log_regex: str = None) -> pd.DataFrame:
    """
    Args:
        fname (str): filename of log file.
        log_regex (str): Regular expression for identifying step information.
            Defaults to '\n(Step[^\n]+\n[^A-Z]+)(?:Loop time)'

    Returns:
        df_log (pandas.DataFrame)
    """
    log_regex = log_regex or '\n(Step[^\n]+\n[^A-Z]+)(?:Loop time)'
    log_blocks = []
    with open(fname, 'r') as f:
        text = f.read()
        for text_block in re.compile(log_regex).findall(text):
            buffer = io.StringIO(text_block)
            df = pd.read_csv(buffer, delim_whitespace=True)
            log_blocks.append(df)
    df_log = pd.concat(log_blocks, ignore_index=True)
    df_log = df_log[~df_log.duplicated()]
    return df_log


def parse_lammps_dump(fname: str,
                      lammps_aliases: Dict[int, str],
                      timesteps: List[int] = None
                      ) -> pd.Series:
    """
    Read LAMMPS text dump file. Expects the following items in the
    thermo_style: id type x y z

    Other items, such as fx and custom computes,
    are added via ase.Atoms.new_array().

    Compatible with large files because the function reads line-by-line
    and, optionally, saves only specified timesteps.

    TODO: refactor to break up into smaller, reusable functions

    Args:
        fname (str): filename of dump file.
        lammps_aliases (dict): map of LAMMSPS type to species.
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
                                          lammps_aliases=lammps_aliases)
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
                if not line:
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


def read_vasp_pressure(path: str) -> float:
    """Utility for reading external pressure (kbar) from PSTRESS INCAR tag.
    Used for extracting energy from VASP enthalpy (H = E + PV)"""
    fname_incar = os.path.join(path, "INCAR")
    fname_outcar = os.path.join(path, "OUTCAR")
    fname_vasprun = os.path.join(path, "vasprun.xml")

    pstress = None
    for fname in [fname_incar, fname_outcar, fname_vasprun]:
        print(fname)
        if os.path.isfile(fname):
            with open(fname, "r") as f:
                line = f.readline()
                while line:
                    if "PSTRESS" in line:
                        pstress = float(re.sub('[^0-9\\.]', '', line))
                        break
                    line = f.readline()
        if isinstance(pstress, float):
            break
    if pstress is None:
        return 0.0
    else:
        external_pressure = pstress * 1e-22 / (1.602176634e-19)
        return external_pressure


def identify_paths(experiment_path: str = ".",
                   filename: str = None,
                   filename_pattern:str = None
                   ) -> List[str]:
    """
    Args:
        experiment_path (str): directory in which to search, recursively.
            Default "."
        filename (str): single filename.
        filename_pattern (str): glob pattern e.g. "*.xyz" to search.

    Returns:
        data_paths (list)
    """
    data_paths = []
    if filename is not None:
        if os.path.isfile(filename):
            data_paths.append(filename)
        elif os.path.isfile(os.path.join(experiment_path, filename)):
            data_paths.append(filename)
    if filename_pattern is not None:
        for directory, folders, files in os.walk(experiment_path):
            for filename in files:
                 if fnmatch.fnmatch(filename, filename_pattern):
                    path = os.path.join(directory, filename)
                    data_paths.append(path)
    return data_paths


def parse_with_subsampling(data_paths: List[str],
                           data_coordinator: DataCoordinator,
                           max_samples: int = 100,
                           min_diff: float = 1e-3,
                           vasp_pressure: bool = False,
                           lammps_log: str = None,
                           lammps_aliases: Dict[int, str] = None,
                           verbose: bool = True):
    """
    TODO: refactor to break up into smaller, reusable functions

    Args:
        data_paths (list)
        data_coordinator (DataCoordinator)
        max_samples (int): maximum number of samples taken per provided path.
            Default: 100
        min_diff (float): minimum energy difference between consecutive samples
            in eV. Default: 1e-3
        energy_key (str): column name for energies, default "energy".
        vasp_pressure (bool): whether to search for pressure and apply an
            energy correction of Pressure * Volume term (H = E + PV).
        lammps_log (str): optional name of lammps log, if applicable.
        lammps_aliases (dict): map of LAMMPS type to species.
        verbose (bool, int): verbosity level.
    """
    common_prefix = os.path.commonprefix(data_paths)
    common_path = os.path.dirname(common_prefix)
    counter = 0

    energy_key = data_coordinator.energy_key
    size_key = data_coordinator.size_key

    for data_path in data_paths:
        prefix = data_path[len(common_path):]
        prefix = prefix.replace("/", "-")
        if prefix[0] == "-":
            prefix = prefix[1:]

        try:
            if lammps_log is not None:
                vasp_pressure = False
                lammps_path, dump_fname = os.path.split(data_path)

                df = data_coordinator.dataframe_from_lammps_run(
                    lammps_path, lammps_aliases, prefix=prefix, load=False,
                    log_fname=lammps_log, dump_fname=dump_fname,
                    column_subs={"TotEng": "energy"})
            else:
                df = data_coordinator.dataframe_from_trajectory(data_path,
                                                                prefix=prefix,
                                                                load=False)
        except ValueError:
            continue
        if len(df) == 0:
            continue
        energy_list = df[energy_key].values / df[size_key].values

        if max_samples > 0:
            subsamples = subsample.farthest_point_sampling(energy_list,
                                                           max_samples=max_samples,
                                                           min_diff=min_diff)
        else:
            subsamples = np.arange(len(energy_list))
        if verbose >= 2:
            print("{}/{} samples taken from {}.".format(len(subsamples),
                                                        len(energy_list),
                                                        prefix))
        counter += len(subsamples)
        if verbose >= 1:
            print("Total: {} samples parsed.".format(counter))
        df = df.iloc[np.sort(subsamples)]

        if vasp_pressure:
            vasp_path = os.path.dirname(data_path)
            external_pressure = read_vasp_pressure(vasp_path)
            if external_pressure != 0:
                volumes = [geom.get_volume() for geom in df['geometry'].values]
                corrections = np.multiply(volumes, external_pressure)
                df[energy_key] = np.subtract(df['energy'], corrections)
            if verbose >= 1:
                line = "External pressure correction: {} kbar."
                print(line.format(external_pressure))
        data_coordinator.load_dataframe(df, prefix=prefix)


def cache_data(data_coordinator: DataCoordinator,
               filename: str,
               energy_key: str = 'energy',
               serial: bool = False):
    """
    Save dataframe from data_coordinator as ase Database.

    Args:
        data_coordinator (DataCoordinator)
        filename (str)
        energy_key (str): column name for energies, default "energy".
        serial (bool)
    """
    append = os.path.isfile(filename)

    df_data = data_coordinator.consolidate()
    geometries = df_data['geometry']

    with ase_db.connect(filename, append=append, serial=serial) as database:
        for name, geom in geometries.iteritems():
            energy = geom.info[energy_key]
            forces = np.vstack([geom.arrays['fx'],
                                geom.arrays['fy'],
                                geom.arrays['fz']]).T
            geom_info = {k: geom.info[k] for k in geom.info
                         if (isinstance(geom.info[k],
                                        (int, float, str, np.floating))
                             and k not in db_core.reserved_keys)}
            geom = geom.copy()
            calc = singlepoint.SinglePointCalculator(geom,
                                                     energy=energy,
                                                     forces=forces)
            geom.calc = calc
            database.write(geom,
                           id=None,
                           key_value_pairs=geom_info,
                           row_name=name)


def analyze_hdf_tables(filename: str) -> Tuple[int, int, List, Dict]:
    """Read hdf5 file and analyze table names and lengths"""
    with tables.open_file(filename, mode="r") as h5file:
        chunk_lengths = {}
        paths = [group._v_name
                 for group in h5file.list_nodes("/")]

        for path in paths:
            table = h5file.get_node("/" + path, "axis0")
            chunk_lengths[path] = table.nrows
    n_chunks = len(chunk_lengths)
    n_entries = int(np.sum([v for v in chunk_lengths.values()]))
    chunk_names = sorted(paths)
    return n_chunks, n_entries, chunk_names, chunk_lengths


def dataframe_batch_loader(filename, table_names):
    for table_name in table_names:
        df = pd.read_hdf(filename, table_name)
        yield df


def resolve_name_conflict(path: str) -> int:
    """Simple renaming by incrementing an integer preceding file extension."""
    if os.path.isfile(path):
        i = 0
        while True:
            stem, ext = os.path.splitext(path)
            backup_path = stem + "." + str(i) + ext
            if not os.path.isfile(backup_path):
                os.rename(path, backup_path)
                break
            i += 1
    return i
