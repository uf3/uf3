import warnings

import numpy as np
import pandas as pd

from uf3.representation import knots
from uf3.representation import distances
from uf3.representation.bspline import evaluate_bspline
from uf3.representation.bspline import compute_force_bsplines
from uf3.data import geometry
from uf3.util import parallel


class BasisProcessor2B:
    """
    -Manage knot-related logic for pair interactions
    -Generate energy/force features
    -Arrange features into DataFrame
    -Process DataFrame into tuples of (x, y, weight)
    """
    def __init__(self,
                 chemical_system,
                 knot_spacing='lammps',
                 name='ij'):
        """
        Args:
            chemical_system (uf3.data.composition.ChemicalSystem)
            knot_spacing (str): "lammps" for knot spacing by r^2
                or "linear" for uniform spacing.
            name (str): prefix for feature columns.
        """
        self.chemical_system = chemical_system
        self.knot_spacing = knot_spacing
        if knot_spacing == 'lammps':
            knot_function = knots.generate_lammps_knots
        elif knot_spacing == 'linear':
            knot_function = knots.generate_uniform_knots
        else:
            raise ValueError('Invalid value of knot_spacing:', knot_spacing)
        interactions_map = chemical_system.interactions_map
        r_min_map = chemical_system.r_min_map
        r_max_map = chemical_system.r_max_map
        resolution_map = chemical_system.resolution_map
        self.r_cut = max(list(r_max_map.values()))  # supercell cutoff
        # compute knots
        self.knots_map = {}
        for pair in interactions_map[2]:
            r_min = r_min_map[pair]
            r_max = r_max_map[pair]
            n_intervals = resolution_map[pair]
            self.knots_map[pair] = knot_function(r_min, r_max, n_intervals)
        self.knot_subintervals = {pair: knots.get_knot_subintervals(knot_seq)
                                  for pair, knot_seq in self.knots_map.items()}
        # generate column labels
        self.n_features = sum([n_intervals + 3 for n_intervals
                               in resolution_map.values()])
        self.columns = ['{}_{}'.format(name, i)
                        for i in range(self.n_features)]
        self.columns.insert(0, 'y')
        self.columns.extend(['n_{}'.format(el) for el
                             in self.chemical_system.element_list])

    @staticmethod
    def from_config(chemical_system, config):
        """Instantiate from configuration dictionary"""
        keys = ['knot_spacing', 'name']
        config = {k: v for k, v in config.items() if k in keys}
        return BasisProcessor2B(chemical_system,
                                **config)

    def get_regularizer_sizes(self):
        """Get sizes of regularizers: two-body and one-body terms"""
        subdivisions = [n_intervals + 3 for n_intervals
                        in self.chemical_system.resolution_map.values()]
        subdivisions.append(len(self.chemical_system.element_list))
        return subdivisions

    def evaluate(self, df_data, data_coordinator, progress_bar=True):
        """
        Process standard dataframe to generate representation features
        and arrange into processed dataframe. Operates in serial by default.

        Args:
            df_data (pd.DataFrame): standard dataframe with columns
                [atoms_key, energy_key, fx, fy, fz]
            data_coordinator (uf3.data.io.DataCoordinator)
            progress_bar (bool)

        Returns:
            df_features (pd.DataFrame): processed dataframe with columns
                [y, {name}_0, ..., {name}_x, n_A, ..., n_Z]
                corresponding to target vector, pair-distance representation
                features, and composition (one-body) features.
        """
        atoms_key = data_coordinator.atoms_key
        energy_key = data_coordinator.energy_key
        eval_map = {}
        header = df_data.columns
        column_positions = {}
        for key in [atoms_key, energy_key, 'fx', 'fy', 'fz']:
            if key in header:
                column_positions[key] = header.get_loc(key) + 1

        if progress_bar:
            row_generator = parallel.progress_iter(df_data.itertuples(name=None),
                                                   total=len(df_data))
        else:
            row_generator = df_data.itertuples(name=None)

        for row in row_generator:
            # iterate over rows without modification.
            name = row[0]
            geom = row[column_positions[atoms_key]]
            energy = None
            forces = None
            if energy_key in column_positions:
                energy = row[column_positions[energy_key]]
            if 'fx' in column_positions:
                forces = [row[column_positions[component]]
                          for component in ['fx', 'fy', 'fz']]
            geom_features = self.evaluate_configuration(geom,
                                                        name,
                                                        energy,
                                                        forces,
                                                        energy_key)
            eval_map.update(geom_features)
        df_features = self.arrange_features_dataframe(eval_map)
        return df_features

    def evaluate_parallel(self, df_data, data_coordinator, client, n_jobs=2):
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
        if n_jobs < 1:
            warnings.warn("Processing in serial.", RuntimeWarning)
            df_features = self.evaluate(df_data, data_coordinator)
            return df_features
        batches = parallel.split_dataframe(df_data, n_jobs)
        future_list = parallel.batch_submit(self.evaluate,
                                            batches,
                                            client,
                                            data_coordinator=data_coordinator)
        df_features = parallel.gather_and_merge(future_list,
                                                client=client,
                                                cancel=True)
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

    def get_training_tuples(self, df_features, kappa, data_coordinator):
        """
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
        energy_key = data_coordinator.energy_key
        x, y, w = dataframe_to_training_tuples(df_features,
                                               kappa=kappa,
                                               energy_key=energy_key)
        return x, y, w

    def evaluate_configuration(self,
                               geom,
                               name=None,
                               energy=None,
                               forces=None,
                               energy_key="energy"):
        """
        Generate feature vector(s) for learning energy and/or forces
            of one configuration.

        Args:
            geom (ase.Atoms): configuration of interest.
            name (str): if specified, keys in returned dictionary are tuples
                {(name, 'e'), (name, 'fx'), ...{ instead of {'e', 'fx', ...}
            energy (float): energy of configuration (optional).
            forces (list, np.ndarray): array containing force components
                fx, fy, fz for each atom. Expected shape is (n_atoms, 3).
            energy_key (str): column name for energies, default "energy".

        Returns:
            eval_map (dict): map of energy/force keys to fixed-length
                feature vectors. If forces and the energy are both provided,
                the dictionary will contain 3N + 1 entries.
        """
        eval_map = {}
        n_atoms = len(geom)
        if any(geom.pbc):
            supercell = geometry.get_supercell(geom, r_cut=self.r_cut)
        else:
            supercell = geom
        if energy is not None:  # compute energy features
            vector = self.get_energy_features(geom, supercell)
            if name is not None:
                key = (name, energy_key)
            else:
                key = energy_key
            eval_map[key] = np.insert(vector, 0, energy)
        if forces is not None:  # compute force features
            vectors = self.get_force_features(geom, supercell)
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

    def get_energy_features(self, geom, supercell=None):
        """
        Generate feature vector for learning energy of one configuration.

        Args:
            geom (ase.Atoms): unit cell or configuration of interest.
            supercell (ase.Atoms): optional ase.Atoms output of get_supercell
                used to account for atoms in periodic images.

        Returns:
            vector (np.ndarray): vector of features.
        """
        if supercell is None:
            supercell = geom
        pair_tuples = self.chemical_system.interactions_map[2]
        r_min_map = self.chemical_system.r_min_map
        r_max_map = self.chemical_system.r_max_map
        distances_map = distances.distances_by_interaction(geom,
                                                           pair_tuples,
                                                           r_min_map,
                                                           r_max_map,
                                                           supercell=supercell)
        feature_map = {}
        for pair in pair_tuples:
            knot_subintervals = self.knot_subintervals[pair]
            features = evaluate_bspline(distances_map[pair],
                                        knot_subintervals)
            feature_map[pair] = features
        feature_vector = flatten_by_interactions(feature_map,
                                                 pair_tuples)
        comp = self.chemical_system.get_composition_tuple(geom)
        vector = np.concatenate([feature_vector, comp])
        return vector

    def get_force_features(self, geom, supercell=None):
        """
        Generate feature vectors for learning forces of one configuration.
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
        pair_tuples = self.chemical_system.interactions_map[2]
        r_min_map = self.chemical_system.r_min_map
        r_max_map = self.chemical_system.r_max_map
        deriv_results = distances.derivatives_by_interaction(geom,
                                                             supercell,
                                                             pair_tuples,
                                                             r_min_map,
                                                             r_max_map)
        distance_map, derivative_map = deriv_results
        feature_map = {}
        for pair in pair_tuples:
            knot_subintervals = self.knot_subintervals[pair]
            features = compute_force_bsplines(derivative_map[pair],
                                              distance_map[pair],
                                              knot_subintervals)
            feature_map[pair] = features
        feature_array = flatten_by_interactions(feature_map,
                                                pair_tuples)
        comp_array = np.zeros((len(feature_array),
                              3,
                              len(self.chemical_system.element_list)))
        feature_array = np.concatenate([feature_array, comp_array], axis=2)
        return feature_array


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

    Returns:
        x (np.ndarray): features for machine learning.
        y (np.ndarray): target vector.
        w (np.ndarray): weight vector for machine learning.
    """
    if kappa < 0 or kappa > 1:
        raise ValueError("Invalid domain for kappa weighting parameter.")
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