"""
This module provides the BasisFeaturizer class for computing
energy and force features, using bspline.BsplineBasis, from DataFrames
containing ase.Atoms configurations.
"""

import os
import warnings
import sqlite3
import numpy as np
import pandas as pd
from uf3.representation import distances
from uf3.representation import angles
from uf3.representation import bspline
from uf3.data import io
from uf3.data import geometry
from uf3.util import parallel


class BasisFeaturizer:
    """
    -Manage knot-related logic for pair interactions
    -Generate energy/force features
    -Arrange features into DataFrame
    -Process DataFrame into tuples of (x, y, weight)
    """
    def __init__(self, bspline_config, fit_forces=True, prefix='x'):
        """
        Args:
            bspline_config (uf3.representation.bspline.BsplineConfig)
            fit_forces (bool): whether to generate force features.
            prefix (str): prefix for feature columns.
        """
        self.bspline_config = bspline_config
        self.fit_forces = fit_forces
        self.prefix = prefix

        # generate column labels
        self.columns = self.bspline_config.get_column_names()

    @property
    def chemical_system(self):
        return self.bspline_config.chemical_system

    @property
    def degree(self):
        return self.chemical_system.degree

    @property
    def element_list(self):
        return self.chemical_system.element_list

    @property
    def interactions_map(self):
        return self.chemical_system.interactions_map

    @property
    def r_min_map(self):
        return self.bspline_config.r_min_map

    @property
    def r_max_map(self):
        return self.bspline_config.r_max_map

    @property
    def resolution_map(self):
        return self.bspline_config.resolution_map

    @property
    def r_cut(self):
        return self.bspline_config.r_cut

    @property
    def knots_map(self):
        return self.bspline_config.knots_map

    @property
    def knot_subintervals(self):
        return self.bspline_config.knot_subintervals

    @property
    def basis_functions(self):
        return self.bspline_config.basis_functions

    @property
    def partition_sizes(self):
        return self.bspline_config.partition_sizes

    @property
    def interaction_hashes(self):
        return self.chemical_system.interaction_hashes

    @property
    def leading_trim(self):
        return self.bspline_config.leading_trim

    @property
    def trailing_trim(self):
        return self.bspline_config.trailing_trim

    @staticmethod
    def from_config(bspline_config, config):
        """Instantiate from configuration dictionary"""
        keys = ['knot_spacing',
                'prefix',
                'fit_forces']
        config = {k: v for k, v in config.items() if k in keys}
        return BasisFeaturizer(bspline_config, **config)

    def __repr__(self):
        summary = ["BasisFeaturizer:",
                   f"    Fit forces: {self.fit_forces}",
                   f"    Column prefix: {self.prefix}",
                   self.bspline_config.__repr__()
                   ]
        return "\n".join(summary)

    def __str__(self):
        return self.__repr__()

    def evaluate(self,
                 df_data,
                 atoms_key="geometry",
                 energy_key="energy",
                 progress="bar"):
        """
        Process standard dataframe to generate representation features
        and arrange into processed dataframe. Operates in serial by default.

        Args:
            df_data (pd.DataFrame): standard dataframe with columns
                [atoms_key, energy_key, fx, fy, fz]
            atoms_key (str)
            energy_key (str)
            progress (str, None): style of progress counter.

        Returns:
            df_features (pd.DataFrame): processed dataframe with columns
                [y, {name}_0, ..., {name}_x, n_A, ..., n_Z]
                corresponding to target vector, pair-distance representation
                features, and composition (one-body) features.
        """
        eval_map = {}
        header = df_data.columns
        column_positions = {}
        for key in [atoms_key, energy_key, 'fx', 'fy', 'fz']:
            if key in header:
                column_positions[key] = header.get_loc(key) + 1

        row_gener = parallel.progress_iter(df_data.itertuples(name=None),
                                           total=len(df_data),
                                           style=progress)

        for row in row_gener:
            # iterate over rows without modification.
            name = row[0]
            geom = row[column_positions[atoms_key]]
            energy = None
            forces = None
            if energy_key in column_positions:
                energy = row[column_positions[energy_key]]
            if 'fx' in column_positions and self.fit_forces:
                forces = [row[column_positions[component]]
                          for component in ['fx', 'fy', 'fz']]
                if np.any(np.isnan(forces)):
                    forces = None  # invalid forces
            geom_features = self.evaluate_configuration(geom,
                                                        name,
                                                        energy,
                                                        forces,
                                                        energy_key)
            eval_map.update(geom_features)
        df_features = self.arrange_features_dataframe(eval_map)
        return df_features

    def arrange_features_dataframe(self, eval_map):
        """
        Args:
            eval_map (dict): map of energy/force keys to fixed-length
                feature vectors. If forces and the energy are both provided,
                the dictionary will contain 3N + 1 entries.

        Returns:
            df_features (pd.DataFrame): processed dataframe with columns
                [y, {name}_0, ..., {name}_x, n_A, ..., n_Z]
                corresponding to target vector, pair-distance representation
                features, and composition (one-body) features.
        """
        df_features = pd.DataFrame.from_dict(eval_map,
                                             orient='index',
                                             columns=self.columns)
        index = pd.MultiIndex.from_tuples(df_features.index)
        df_features = df_features.set_index(index)
        return df_features

    def evaluate_parallel(self,
                          df_data,
                          client,
                          atoms_key="geometry",
                          energy_key="energy",
                          n_jobs=2,
                          shuffle=True,
                          progress="bar"):
        """
        Process standard dataframe to generate representation features
        and arrange into processed dataframe. Operates in serial by default.

        Args:
            df_data (pd.DataFrame): standard dataframe with columns
                [atoms_key, energy_key, fx, fy, fz]
            data_coordinator (uf3.data.io.DataCoordinator)
            n_jobs (int): number of parallel jobs to submit.
            client (concurrent.futures.Executor, dask.distributed.Client)

        Returns:
            df_features (pd.DataFrame): processed dataframe with columns
                [y, {name}_0, ..., {name}_x, n_A, ..., n_Z]
                corresponding to target vector, pair-distance representation
                features, and composition (one-body) features.
        """
        if n_jobs < 2:
            warnings.warn("Processing in serial.", RuntimeWarning)
            df_features = self.evaluate(df_data,
                                        atoms_key=atoms_key,
                                        energy_key=energy_key)
            return df_features
        if shuffle:
            shuffled_idx = np.arange(len(df_data))
            np.random.shuffle(shuffled_idx)
            batches = parallel.split_dataframe(df_data.iloc[shuffled_idx],
                                               n_jobs)
        else:
            batches = parallel.split_dataframe(df_data, n_jobs)
        try:
            batches = [client.scatter(batch) for batch in batches]
        except AttributeError:
            pass
        future_list = parallel.batch_submit(self.evaluate,
                                            batches,
                                            client,
                                            atoms_key=atoms_key,
                                            energy_key=energy_key,
                                            progress=False)
        df_features = parallel.gather_and_merge(future_list,
                                                client=client,
                                                cancel=True,
                                                progress=progress)
        df_features = df_features.loc[df_data.index, :]
        try:
            for batch in batches:
                client.cancel(batch)
        except AttributeError:
            pass
        return df_features

    def batched_to_hdf(self,
                       filename,
                       df_data,
                       client,
                       n_jobs=16,
                       batch_size=50,
                       progress="bar",
                       table_template="features_{}",
                       **kwargs):
        idx_all = np.arange(len(df_data))
        idx_splits = idx_all[batch_size::batch_size]
        idx_batches = np.array_split(idx_all, idx_splits)
        n_batches = len(idx_batches)
        idx_magnitude = np.ceil(np.log10(n_batches) + 0.1).astype(int)
        idx_magnitude = max(idx_magnitude, 3)

        if os.path.isfile(filename):
            n_chunks, _, chunk_names, _ = io.analyze_hdf_tables(filename)
            warnings.warn(f"File already exists: contains {n_chunks} chunks.",
                          RuntimeWarning)
        else:
            chunk_names = []

        for j, idx_batch in parallel.progress_iter(enumerate(idx_batches),
                                                   total=len(idx_batches),
                                                   style=progress):
            table_name = table_template.format(str(j).rjust(idx_magnitude,
                                                            "0"))
            if table_name in chunk_names:
                continue
            kwargs["progress"] = False
            kwargs["n_jobs"] = n_jobs
            df_features = self.evaluate_parallel(df_data.iloc[idx_batch],
                                                 client,
                                                 **kwargs)
            save_feature_db(df_features, filename, table_name=table_name)

    def evaluate_configuration(self,
                               geom,
                               name=None,
                               energy=None,
                               forces=None,
                               energy_key="energy"):
        """
        Generate feature vector(s) for learning energy and/or forces
            of one configuration.

        TODO: refactor to break up into smaller, reusable functions

        Args:
            geom (ase.Atoms): configuration of interest.
            name (str): if specified, keys in returned dictionary are tuples
                {(name, 'e'), (name, 'fx'), ...{ instead of {'e', 'fx', ...}
            energy (float): energy of configuration (optional).
            forces (list, np.ndarray): array containing force components
                fx, fy, fz for each atom. Expected shape is (3, n_atoms).
            energy_key (str): column name for energies, default "energy".

        Returns:
            eval_map (dict): map of energy/force keys to fixed-length
                feature vectors. If forces and the energy are both provided,
                the dictionary will contain 3N + 1 entries.
        """
        eval_map = {}
        n_atoms = len(geom)
        symbols = set(geom.get_chemical_symbols())
        # check for undefined elements
        invalid_set = symbols.difference(self.element_list)
        if len(invalid_set) > 0:
            invalid_set = ', '.join(invalid_set)
            warning_str = "Invalid elements: {}".format(invalid_set)
            if name is not None:
                warning_str += " in configuration "+name
            warnings.warn(warning_str, RuntimeWarning)
            return dict()
        if any(geom.pbc):  # generate supercell if necessary
            supercell = geometry.get_supercell(geom, r_cut=self.r_cut)
        else:
            supercell = geom
        if energy is not None:  # compute energy features
            vector_1B = self.chemical_system.get_composition_tuple(geom)
            vector_2B = self.featurize_energy_2B(geom, supercell)
            vector = np.concatenate([vector_1B, vector_2B])
            if self.degree > 2:
                vector_3B = self.featurize_energy_3B(geom, supercell)
                vector = np.concatenate([vector, vector_3B])
            if name is not None:
                key = (name, energy_key)
            else:
                key = energy_key
            eval_map[key] = np.insert(vector, 0, energy)
        if forces is not None:  # compute force features
            vectors_1B = np.zeros((len(geom),
                                   3,
                                   len(self.element_list)))
            vectors_2B = self.featurize_force_2B(geom, supercell)
            vectors = np.concatenate([vectors_1B, vectors_2B],
                                     axis=2)
            if self.degree > 2:
                vectors_3B = self.featurize_force_3B(geom, supercell)
                vectors = np.concatenate([vectors, vectors_3B], axis=2)
            for j, component in enumerate(['fx', 'fy', 'fz']):
                for i in range(n_atoms):
                    vector = vectors[i, j, :]
                    vector = np.insert(vector, 0, forces[j][i])
                    atom_index = component + '_' + str(i)
                    if name is not None:
                        key = (name, atom_index)
                    else:
                        key = atom_index
                    eval_map[key] = vector
        return eval_map

    def featurize_energy_2B(self, geom, supercell=None):
        """
        Generate 2B feature vector for learning energy of one configuration.

        Args:
            geom (ase.Atoms): unit cell or configuration of interest.
            supercell (ase.Atoms): optional ase.Atoms output of get_supercell
                used to account for atoms in periodic images.

        Returns:
            vector (np.ndarray): vector of features.
        """
        if supercell is None:
            supercell = geom
        pair_tuples = self.interactions_map[2]
        distances_map = distances.distances_by_interaction(geom,
                                                           pair_tuples,
                                                           self.r_min_map,
                                                           self.r_max_map,
                                                           supercell=supercell)
        feature_map = {}
        for pair in pair_tuples:
            basis_function = self.basis_functions[pair]
            features = bspline.evaluate_basis_functions(
                distances_map[pair],
                basis_function,
                n_lead=self.leading_trim[2],
                n_trail=self.trailing_trim[2])
            feature_map[pair] = features
        feature_vector = flatten_by_interactions(feature_map,
                                                 pair_tuples)
        return feature_vector

    def featurize_force_2B(self, geom, supercell=None):
        """
        Generate 2B feature vectors for learning forces of one configuration.
        Args:
            geom (ase.Atoms): unit cell or configuration of interest.
            supercell (ase.Atoms): optional ase.Atoms output of get_supercell
                used to account for atoms in periodic images.

        Returns:
            feature_array (np.ndarray): feature vectors arranged in
                array of shape (n_atoms, n_force_components, n_features).
        """
        if supercell is None:
            supercell = geom
        pair_tuples = self.interactions_map[2]
        deriv_results = distances.derivatives_by_interaction(geom,
                                                             pair_tuples,
                                                             self.r_cut,
                                                             self.r_min_map,
                                                             self.r_max_map,
                                                             supercell)
        distance_map, derivative_map = deriv_results
        feature_map = {}
        for pair in pair_tuples:
            basis_functions = self.basis_functions[pair]
            knot_sequence = self.knots_map[pair]
            features = bspline.featurize_force_2B(
                basis_functions,
                distance_map[pair],
                derivative_map[pair],
                knot_sequence,
                n_lead=self.leading_trim[2],
                n_trail=self.trailing_trim[2])
            feature_map[pair] = features
        feature_array = flatten_by_interactions(feature_map,
                                                pair_tuples)
        return feature_array

    def featurize_energy_3B(self, geom, supercell=None):
        """
        Generate 3B feature vector for learning energy of one configuration.

        Args:
            geom (ase.Atoms): unit cell or configuration of interest.
            supercell (ase.Atoms): optional ase.Atoms output of get_supercell
                used to account for atoms in periodic images.

        Returns:
            vector (np.ndarray): vector of features.
        """
        if supercell is None:
            supercell = geom
        trio_list = self.interactions_map[3]
        knot_sets = [self.knots_map[trio] for trio in trio_list]
        basis_functions = [self.basis_functions[trio] for trio in trio_list]
        hashes = self.interaction_hashes[3]
        grids = angles.featurize_energy_3b(geom,
                                           knot_sets,
                                           basis_functions,
                                           hashes,
                                           supercell=supercell,
                                           n_lead=self.leading_trim[3],
                                           n_trail=self.trailing_trim[3])
        vectors = []
        for i, trio in enumerate(trio_list):
            value_grid = grids[i]
            vector = self.bspline_config.compress_3B(value_grid, trio)
            vectors.append(vector)
        vector = np.concatenate(vectors)
        return vector

    def featurize_force_3B(self, geom, supercell=None):
        """
        Generate 3B feature vectors for learning forces of one configuration.

        Args:
            geom (ase.Atoms): unit cell or configuration of interest.
            supercell (ase.Atoms): optional ase.Atoms output of get_supercell
                used to account for atoms in periodic images.

        Returns:
            feature_array (np.ndarray): feature vectors arranged in
                array of shape (n_atoms, n_force_components, n_features).
        """
        if supercell is None:
            supercell = geom
        trio_list = self.interactions_map[3]
        knot_sets = [self.knots_map[trio] for trio in trio_list]
        basis_functions = [self.basis_functions[trio] for trio in trio_list]
        hashes = self.interaction_hashes[3]
        grids = angles.featurize_force_3b(geom,
                                          knot_sets,
                                          basis_functions,
                                          hashes,
                                          supercell=supercell,
                                          n_lead=self.leading_trim[3],
                                          n_trail=self.trailing_trim[3])
        blocks = []
        for i, trio in enumerate(trio_list):
            values = grids[i]
            block = [[self.bspline_config.compress_3B(grid, trio)
                      for grid in atom_set]
                     for atom_set in values]
            blocks.append(block)
        return np.concatenate(blocks, axis=-1)

    def get_training_tuples(self, df_features, kappa, data_coordinator):
        """
        TODO: Remove (deprecated)

        Weights are generated by normalizing energy and force entries by the
            respective sample standard deviations as well as the relative
            number of entries per type. Weights are further modified by kappa,
            which controls the relative weighting between energy and force
            errors. A value of 0 corresponds to force-training,
            while a value of 1 corresponds to energy-training.

        Args:
            df_features (pd.DataFrame): dataframe with target vector (y) as the
                first column and feature vectors (x) as remaining columns.
            kappa (float): energy-force weighting parameter between 0 and 1.
            data_coordinator (uf3.data.io.DataCoordinator)

        Returns:
            x (np.ndarray): features for machine learning.
            y (np.ndarray): target vector.
            w (np.ndarray): weight vector for machine learning.
        """
        warnings.warn("get_training_tuples() is deprecated.", DeprecationWarning)
        energy_key = data_coordinator.energy_key
        x, y, w = dataframe_to_training_tuples(df_features,
                                               kappa=kappa,
                                               energy_key=energy_key)
        return x, y, w


def save_feature_db(dataframe, filename, table_name='features'):
    """
    Save dataframe with sqlite.

    Args:
        dataframe (pd.DataFrame)
        filename (str)
        table_name (str): default "features".
    """
    dataframe.to_hdf(filename, table_name, mode="a", format='fixed')


def load_feature_db(filename, table_name='features'):
    """
    Load dataframe with sqlite.

    Args:
        filename (str)
        table_name (str): default "features".

    Returns:
        dataframe (pd.DataFrame)
    """
    dataframe = pd.read_hdf(filename, table_name)
    return dataframe


def legacy_load_feature_db(filename, table_name):
    # deprecated
    conn = sqlite3.connect(filename)
    dataframe = pd.read_sql_query("SELECT * FROM {};".format(table_name), conn)
    dataframe.set_index(keys=['level_0', 'level_1'], inplace=True)
    conn.close()
    return dataframe


def dataframe_to_training_tuples(df_features,
                                 kappa=0.5,
                                 energy_key='energy'):
    """
    Args:
        df_features (pd.DataFrame): dataframe with target vector (y) as the
            first column and feature vectors (x) as remaining columns.
        kappa (float): energy-force weighting parameter between 0 and 1.
        energy_key (str): key for energy samples, used to slice df_features
            into energies and forces for weight generation.

    TODO: refactor to break up into smaller, reusable functions

    Returns:
        x (np.ndarray): features for machine learning.
        y (np.ndarray): target vector.
        w (np.ndarray): weight vector for machine learning.
    """
    if kappa < 0 or kappa > 1:
        raise ValueError("Invalid domain for kappa weighting parameter.")
    if len(df_features) <= 1:
        raise ValueError(
            f"Not enough samples ({len(df_features)} provided)")
    y_index = df_features.index.get_level_values(-1)
    energy_mask = (y_index == energy_key)
    force_mask = np.logical_not(energy_mask)
    data = df_features.to_numpy()
    y = data[:, 0]
    x = data[:, 1:]
    energy_values = y[energy_mask]
    force_values = y[force_mask]
    # compute metrics for weighting
    energy_std = np.std(energy_values)
    force_std = np.std(force_values)
    n_energy = len(energy_values)
    n_forces = len(force_values)
    # generate weights based on sample standard deviation and frequency.
    energy_weights = np.ones(n_energy) / energy_std / n_energy * kappa
    force_weights = np.ones(n_forces) / force_std / n_forces * (1 - kappa)
    w = np.zeros(n_energy + n_forces)
    w[energy_mask] = energy_weights
    w[force_mask] = force_weights
    return x, y, w


def flatten_by_interactions(vector_map, pair_tuples):
    """
    Args:
        vector_map (dict): map of vector per interaction.
        pair_tuples (list): list of pair interactions
            e.g. [(A-A), (A-B), (A-C), (B-B), ...)]

    Returns:
        (np.ndarray): concatenated result of joining vector_map
            in order of occurrence in pair_tuples.
    """
    return np.concatenate([vector_map[pair] for pair in pair_tuples], axis=-1)
