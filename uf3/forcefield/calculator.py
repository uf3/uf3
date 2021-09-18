from typing import List, Dict, Collection, Tuple, Any
import os
import time
import warnings
import numpy as np
from scipy import interpolate
import ase
from ase import optimize as ase_optim
from ase import constraints as ase_constraints
from ase.calculators import calculator as ase_calc

from uf3.data import geometry
from uf3.representation import distances
from uf3.representation import bspline
from uf3.representation import angles
from uf3.regression import regularize
from uf3.regression import least_squares
from uf3.forcefield.properties import elastic
from uf3.forcefield.properties import phonon
import ndsplines


class UFCalculator(ase_calc.Calculator):
    def __init__(self,
                 bspline_config: bspline.BSplineBasis,
                 model: least_squares.WeightedLinearModel):
        # super().__init__()  # TODO: improve compatibility with ASE protocol
        self.bspline_config = bspline_config
        self.model = model
        self.solutions = coefficients_by_interaction(self.element_list,
                                                     self.interactions_map,
                                                     self.partition_sizes,
                                                     model.coefficients)
        self.pair_potentials = construct_pair_potentials(self.solutions,
                                                         self.bspline_config)
        if self.degree > 2:
            if len(self.element_list) > 1:
                raise ValueError("Multicomponent 3B is not yet implemented.")
            self.trio = self.bspline_config.interactions_map[3][0]
            c_compressed = self.solutions[self.trio]
            c_decompressed = bspline_config.decompress_3B(c_compressed,
                                                          self.trio)
            self.coefficients_3b = angles.symmetrize_3B(c_decompressed)

    @property
    def degree(self):
        return self.bspline_config.degree

    @property
    def element_list(self):
        return self.bspline_config.element_list

    @property
    def interactions_map(self):
        return self.bspline_config.interactions_map

    @property
    def r_min_map(self):
        return self.bspline_config.r_min_map

    @property
    def r_max_map(self):
        return self.bspline_config.r_max_map

    @property
    def r_cut(self):
        return self.bspline_config.r_cut

    @property
    def knot_subintervals(self):
        return self.bspline_config.knot_subintervals

    @property
    def partition_sizes(self):
        return self.bspline_config.partition_sizes

    @property
    def coefficients(self):
        return self.model.coefficients

    @property
    def chemical_system(self):
        return self.bspline_config.chemical_system

    def get_potential_energy(self,
                             atoms: ase.Atoms = None,
                             force_consistent: bool = None
                             ) -> float:
        """Evaluate the total energy of a configuration."""""
        if any(atoms.pbc):
            supercell = geometry.get_supercell(atoms, r_cut=self.r_cut)
        else:
            supercell = atoms
        pair_tuples = self.interactions_map[2]
        distances_map = distances.distances_by_interaction(atoms,
                                                           pair_tuples,
                                                           self.r_min_map,
                                                           self.r_max_map,
                                                           supercell)
        energy = 0
        if force_consistent is not True:
            el_set, el_counts = np.unique(atoms.get_chemical_symbols(),
                                          return_counts=True)
            for el, count in zip(el_set, el_counts):
                energy += self.solutions[el] * count
        for pair in pair_tuples:
            r_min = self.r_min_map[pair]
            r_max = self.r_max_map[pair]
            distance_list = distances_map[pair]
            mask = (distance_list > r_min) & (distance_list < r_max)
            distance_list = distance_list[mask]
            bspline_values = self.pair_potentials[pair](distance_list)
            # energy contribution per distance
            energy += np.sum(bspline_values)
        if self.degree >= 3:
            knot_sequences = self.bspline_config.knots_map[self.trio]
            c_grid = self.coefficients_3b
            energy += evaluate_energy_3b(atoms,
                                         knot_sequences,
                                         c_grid,
                                         supercell)
        return energy

    def get_forces(self, atoms: ase.Atoms = None) -> np.ndarray:
        """Return the forces in a configuration."""
        n_atoms = len(atoms)
        if any(atoms.pbc):
            supercell = geometry.get_supercell(atoms, r_cut=self.r_cut)
        else:
            supercell = atoms
        pair_tuples = self.interactions_map[2]
        deriv_results = distances.derivatives_by_interaction(atoms,
                                                             pair_tuples,
                                                             self.r_cut,
                                                             self.r_min_map,
                                                             self.r_max_map,
                                                             supercell)
        distance_map, derivative_map = deriv_results

        forces = np.zeros((n_atoms, 3))
        for pair in pair_tuples:
            r_min = self.r_min_map[pair]
            r_max = self.r_max_map[pair]
            distance_list = distance_map[pair]
            drij_dr = derivative_map[pair]
            mask = (distance_list > r_min) & (distance_list < r_max)
            distance_list = distance_list[mask]
            # first derivative
            bspline_values = self.pair_potentials[pair](distance_list, nu=1)
            deltas = drij_dr[:, :, mask]  # mask position deltas by distances
            # broadcast multiplication over atomic and cartesian axis dims
            component = np.sum(np.multiply(bspline_values, deltas), axis=-1)
            forces -= component
        if self.degree >= 3:
            knot_sequences = self.bspline_config.knots_map[self.trio]
            c_grid = self.coefficients_3b
            forces += evaluate_forces_3b(atoms,
                                         knot_sequences,
                                         c_grid,
                                         supercell)
        return -forces

    def get_stress(self,
                   atoms: ase.Atoms = None,
                   **kwargs
                   ) -> np.ndarray:
        """Return the (numerical) stress."""
        return self.calculate_numerical_stress(atoms, **kwargs)

    def relax_fmax(self,
                   geom: ase.Atoms,
                   fmax: float = 0.05,
                   relax_cell: bool = True,
                   verbose: bool = False,
                   timeout: float = 60.0,
                   **kwargs
                   ) -> ase.Atoms:
        """Minimize max. force using ASE's QuasiNewton optimizer."""
        geom = geom.copy()
        geom.set_calculator(self)

        if np.all(geom.pbc) and relax_cell:  # periodic boundary conditions
            geom_filter = ase_constraints.ExpCellFilter(geom)
        else:
            geom_filter = geom
        if verbose:
            logfile = '-'  # print to stdout
        else:
            logfile = os.devnull  # suppress output
        t0 = time.time()
        optimizer = ase_optim.BFGSLineSearch(geom_filter,
                                             logfile=logfile,
                                             force_consistent=True,
                                             **kwargs)
        for _ in optimizer.irun(fmax=fmax):
            optimizer.step()
            if (time.time() - t0) > timeout:
                warnings.warn("Relaxation timed out.", RuntimeWarning)
                break
        return geom

    def calculation_required(self, atoms: ase.Atoms, quantities: List) -> bool:
        """Check if a calculation is required."""
        if any([q in quantities for q in ['magmom', 'stress', 'charges']]):
            return True
        if 'energy' in quantities and 'energy' not in atoms.info:
            return True
        if 'force' in quantities and 'fx' not in atoms.arrays:
            return True
        return False

    def get_elastic_constants(self, atoms: ase.Atoms) -> np.ndarray:
        results = elastic.get_elastic_constants(atoms, self)
        return results

    def get_phonon_data(self,
                        atoms: ase.Atoms,
                        n_super: int = 5,
                        disp: float = 0.05,
                        ) -> Tuple[Any, Dict, Dict]:
        results = phonon.compute_phonon_data(atoms,
                                             self,
                                             n_super=n_super,
                                             disp=disp)
        return results


def evaluate_energy_3b(geom: ase.Atoms,
                       knot_sequences: List[np.ndarray],
                       c_grid: np.ndarray,
                       sup_geom: ase.Atoms = None
                       ) -> float:
    # TODO: multicomponent
    if sup_geom is None:
        sup_geom = geom
    spline_evaluator = ndsplines.NDSpline(knot_sequences, c_grid, 3)
    # identify pairs
    dist_matrix, i_where, j_where = angles.identify_ij(geom,
                                                       knot_sequences,
                                                       sup_geom)
    if len(i_where) == 0:
        return 0.0
    # generate valid i, j, k triplets by joining i-j and i-j' pairs
    triplet_groups = angles.generate_triplets(
        i_where, j_where, dist_matrix, knot_sequences)
    energy = 0
    for atom_idx, l, m, n, tuples in triplet_groups:
        triangles = np.vstack([l, m, n]).T
        energy += np.sum(spline_evaluator(triangles))
    return energy


def evaluate_forces_3b(geom: ase.Atoms,
                       knot_sequences: List[np.ndarray],
                       c_grid: np.ndarray,
                       sup_geom: ase.Atoms = None
                       ) -> np.ndarray:
    # TODO: multicomponent
    if sup_geom is None:
        sup_geom = geom
    n_atoms = len(geom)
    spline_evaluator = ndsplines.NDSpline(knot_sequences, c_grid, 3)
    # identify pairs
    coords, dist_matrix, i_where, j_where = angles.identify_ij(
        geom, knot_sequences, sup_geom, square=True)
    triplet_groups = angles.generate_triplets(
        i_where, j_where, dist_matrix, knot_sequences)
    f_accumulate = np.zeros((n_atoms, 3))
    for atom_idx, r_l, r_m, r_n, idx_ijk in triplet_groups:
        drij_dr = distances.compute_direction_cosines(coords,
                                                      dist_matrix,
                                                      idx_ijk[:, 0],
                                                      idx_ijk[:, 1],
                                                      n_atoms)
        drik_dr = distances.compute_direction_cosines(coords,
                                                      dist_matrix,
                                                      idx_ijk[:, 0],
                                                      idx_ijk[:, 2],
                                                      n_atoms)
        drjk_dr = distances.compute_direction_cosines(coords,
                                                      dist_matrix,
                                                      idx_ijk[:, 1],
                                                      idx_ijk[:, 2],
                                                      n_atoms)
        triangles = np.vstack([r_l, r_m, r_n]).T
        val_l = spline_evaluator(triangles, nus=np.array([1, 0, 0]))
        val_m = spline_evaluator(triangles, nus=np.array([0, 1, 0]))
        val_n = spline_evaluator(triangles, nus=np.array([0, 0, 1]))
        f_accumulate += np.dot(drij_dr, val_l)
        f_accumulate += np.dot(drik_dr, val_m)
        f_accumulate += np.dot(drjk_dr, val_n)
    return -f_accumulate


def coefficients_by_interaction(element_list: List,
                                interactions_map: Dict[int, List[Tuple[str]]],
                                partition_sizes: Collection[int],
                                coefficients: List[np.ndarray]
                                ) -> Dict[Tuple[str], np.ndarray]:
    """
    Args:
        element_list (list)
        interactions_map (dict): map of degree to list of interactions
            e.g. {2: [('Ne', 'Ne'), ('Ne', 'Xe'), ...]}
        partition_sizes (list): number of coefficients per section.
        coefficients (list): vector of joined, fit coefficients.

    Returns:
        solutions (dict)
    """
    n_elements = len(element_list)
    split_indices = np.cumsum(partition_sizes)[:-1]
    solutions_list = np.array_split(coefficients,
                                    split_indices)
    solutions = {element: value for element, value
                 in zip(element_list, solutions_list[:n_elements])}
    n_i = len (element_list)
    keys = interactions_map[2] + interactions_map.get(3, [])
    for idx, key in enumerate(keys):
        solutions[key] = solutions_list[n_i + idx]
    return solutions


def construct_pair_potentials(coefficient_sets: Dict[Tuple[str], np.ndarray],
                              bspline_config: bspline.BSplineBasis
                              ) -> Dict[Tuple[str], interpolate.BSpline]:
    """
    Args:
        coefficient_sets (dict): map of pair tuple to coefficient vector.
        bspline_config (bspline.BSplineBasis)

    Returns:
        potentials (dict): map of pair tuple to interpolate.BSpline
    """
    pair_tuples = bspline_config.chemical_system.interactions_map[2]
    potentials = {}
    for pair in pair_tuples:
        knot_sequence = bspline_config.knots_map[pair]
        bspline_curve = interpolate.BSpline(knot_sequence,
                                            coefficient_sets[pair],
                                            3,  # cubic BSpline
                                            extrapolate=False)
        potentials[pair] = bspline_curve
    return potentials


def regenerate_coefficients(x: np.ndarray,
                            y: np.ndarray,
                            knot_sequence: np.ndarray,
                            dy: np.ndarray = None
                            ) -> np.ndarray:
    """legacy function for regenerating coefficients from LAMMPS table."""
    knot_subintervals = bspline.get_knot_subintervals(knot_sequence)
    n_splines = len(knot_subintervals)
    y_features = []
    dy_features = []
    for r in x:
        # loop over samples
        points = np.array([r])
        y_components = []
        dy_components = []
        for idx in range(n_splines):
            # loop over number of basis functions
            b_knots = knot_subintervals[idx]
            bs_l = interpolate.BSpline.basis_element(b_knots,
                                                     extrapolate=False)
            mask = np.logical_and(points >= b_knots[0],
                                  points <= b_knots[4])
            y_components.append(bs_l(points[mask]))  # b-spline value
            dy_components.append(bs_l(points[mask], nu=1))  # derivative value
        y_features.append([np.sum(values) for values in y_components])
        dy_features.append([np.sum(values) for values in dy_components])
    matrix = regularize.get_regularizer_matrix(n_splines,
                                               ridge=1e-6,
                                               curvature=1e-5)
    if dy is not None:
        x = np.vstack([y_features, dy_features])
        y = np.concatenate([y, dy])
    else:
        x = y_features
    coefficients = least_squares.weighted_least_squares(x,
                                                        y,
                                                        regularizer=matrix)
    return coefficients
