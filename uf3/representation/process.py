import numpy as np
import pandas as pd

from uf3.representation import knots, distances
from uf3.representation.bspline import evaluate_bspline, compute_force_bsplines
from uf3.data import geometry


class BasisProcessor2B:
    def __init__(self,
                 chemistry_config,
                 knot_spacing='lammps',
                 name='ij'):
        self.chemistry_config = chemistry_config
        self.knot_spacing = knot_spacing
        if knot_spacing == 'lammps':
            knot_function = knots.generate_lammps_knots
        elif knot_spacing == 'linear':
            knot_function = knots.generate_uniform_knots
        else:
            raise ValueError('Invalid value of knot_spacing:', knot_spacing)
        interactions_map = chemistry_config.interactions_map
        r_min_map = chemistry_config.r_min_map
        r_max_map = chemistry_config.r_max_map
        resolution_map = chemistry_config.resolution_map
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
                             in self.chemistry_config.element_list])

    @staticmethod
    def from_config(chemistry_handler, config):
        """Instantiate from configuration dictionary"""
        keys = ['knot_spacing', 'name']
        config = {k: v for k, v in config.items() if k in keys}
        return BasisProcessor2B(chemistry_handler,
                                **config)

    def get_regularizer_sizes(self):
        """Get sizes of regularizers: two-body and one-body terms"""
        subdivisions = [n_intervals + 3 for n_intervals
                        in self.chemistry_config.resolution_map.values()]
        subdivisions.append(len(self.chemistry_config.element_list))
        return subdivisions

    def evaluate(self, df, data_coordinator, xy_out=True):
        atoms_key = data_coordinator.atoms_key
        energy_key = data_coordinator.energy_key
        eval_map = {}
        header = df.columns
        column_positions = {}
        for key in [atoms_key, energy_key, 'fx', 'fy', 'fz']:
            if key in header:
                column_positions[key] = header.get_loc(key) + 1
        for row in df.itertuples(name=None):
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
        df = pd.DataFrame.from_dict(eval_map,
                                    orient='index',
                                    columns=self.columns)
        if xy_out:
            element_list = self.chemistry_config.element_list
            x, y, u, v = dataframe_to_training_pairs(df, element_list)
            return x, y, u, v
        else:
            return df

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
        pair_tuples = self.chemistry_config.interactions_map[2]
        r_min_map = self.chemistry_config.r_min_map
        r_max_map = self.chemistry_config.r_max_map
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
        comp = self.chemistry_config.get_composition_tuple(geom)
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
        pair_tuples = self.chemistry_config.interactions_map[2]
        r_min_map = self.chemistry_config.r_min_map
        r_max_map = self.chemistry_config.r_max_map
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
                              len(self.chemistry_config.element_list)))
        feature_array = np.concatenate([feature_array, comp_array], axis=2)
        return feature_array


def dataframe_to_training_pairs(df, element_list):
    n_onebody = len(element_list)
    onebody_columns = df.columns[-n_onebody:]
    onebody_sums = np.sum(df[onebody_columns].values, axis=1)
    force_mask = (onebody_sums == 0)
    df_energy = df[~force_mask]
    df_forces = df[force_mask]
    x = df_energy.to_numpy()[:, 1:]
    y = df_energy['y'].values
    u = df_forces.to_numpy()[:, 1:]
    v = df_forces['y'].values
    return x, y, u, v


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