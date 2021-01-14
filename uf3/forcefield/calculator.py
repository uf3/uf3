import os
import time
import warnings
import numpy as np
from scipy import interpolate
from uf3.data import geometry
from uf3.representation import distances
from uf3.representation import bspline
from uf3.regression import least_squares
from ase.calculators import calculator as ase_calc
from ase import optimize as ase_optim
from ase import constraints as ase_constraints


class UFCalculator(ase_calc.Calculator):
    def __init__(self, bspline_config, model):
        """

        Args:
            bspline_config (bspline.BSplineConfig)
            model (least_squares.WeightedLinearModel)
        """
        self.bspline_config = bspline_config
        self.model = model
        self.solutions = coefficients_by_interaction(self.element_list,
                                                     self.interactions_map,
                                                     self.partition_sizes,
                                                     model.coefficients)
        self.pair_potentials = construct_pair_potentials(self.solutions,
                                                         self.bspline_config)

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

    def get_potential_energy(self, atoms=None, force_consistent=None):
        """Evaluate the total energy of a configuration."""""
        n_atoms = len(atoms)
        if any(atoms.pbc):
            supercell = geometry.get_supercell(atoms, r_cut=self.r_cut)
        else:
            supercell = atoms
        pair_tuples = self.interactions_map[2]
        distances_map = distances.distances_by_interaction(atoms,
                                                           pair_tuples,
                                                           self.r_min_map,
                                                           self.r_max_map,
                                                           supercell=supercell)
        energy = 0
        for pair in pair_tuples:
            r_min = self.r_min_map[pair]
            r_max = self.r_max_map[pair]
            distance_list = distances_map[pair]
            mask = (distance_list > r_min) & (distance_list < r_max)
            distance_list = distance_list[mask]
            bspline_values = self.pair_potentials[pair](distance_list)
            # energy contribution per distance
            energy += np.sum(bspline_values)
        return energy

    def get_forces(self, atoms=None):
        """Return the forces in a configuration."""
        n_atoms = len(atoms)
        if any(atoms.pbc):
            supercell = geometry.get_supercell(atoms, r_cut=self.r_cut)
        else:
            supercell = atoms
        pair_tuples = self.interactions_map[2]
        deriv_results = distances.derivatives_by_interaction(atoms,
                                                             supercell,
                                                             pair_tuples,
                                                             self.r_min_map,
                                                             self.r_max_map)
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
            forces += component
        return forces

    def get_stress(self, atoms=None, **kwargs):
        """Return the (numerical) stress."""
        return self.calculate_numerical_stress(atoms, **kwargs)

    def relax_fmax(self, geom, fmax=0.05, verbose=False, timeout=60, **kwargs):
        """Minimize max. force using ASE's QuasiNewton optimizer."""
        geom = geom.copy()
        geom.set_calculator(self)
        cell_filter = ase_constraints.ExpCellFilter(geom)
        if verbose:
            logfile = '-'
        else:
            logfile = os.devnull
        t0 = time.time()
        best_snapshot = geom.copy()
        best_force = np.max(geom.get_forces())
        optimizer = ase_optim.BFGSLineSearch(cell_filter,
                                             logfile=logfile,
                                             **kwargs)
        for _ in optimizer.irun(fmax=fmax):
            print(geom.get_volume(), end=' ')
            optimizer.step()
            print(geom.get_volume())
            i_force = np.max(geom.get_forces())
            if i_force < best_force:
                best_force = i_force
                best_snapshot = geom.copy()
            if (time.time() - t0) > timeout:
                warnings.warn("Relaxation timed out.", RuntimeWarning)
                break
        return best_snapshot

    def calculation_required(self, atoms, quantities):
        """Check if a calculation is required."""
        if any([q in quantities for q in ['magmom', 'stress', 'charges']]):
            return True
        if 'energy' in quantities and 'energy' not in atoms.info:
            return True
        if 'force' in quantities and 'fx' not in atoms.arrays:
            return True
        return False


def coefficients_by_interaction(element_list,
                                interactions_map,
                                partition_sizes,
                                coefficients):
    """
    Args:
        element_list (list)
        interactions_map (dict): map of degree to list of interactions
            e.g. {2: [('Ne', 'Ne'), ('Ne', 'Xe'), ...]}
        partition_sizes (list, np.ndarray): number of coefficients per section.
        coefficients (list, np.ndarray): vector of joined, fit coefficients.

    Returns:

    """
    split_indices = np.cumsum(partition_sizes)[:-1]
    solutions_list = np.array_split(coefficients,
                                    split_indices)
    solutions = {element: value for element, value
                 in zip(element_list, solutions_list[0])}
    for i, pair in enumerate(interactions_map[2]):
        solutions[pair] = solutions_list[i + 1]
    return solutions


def construct_pair_potentials(coefficient_sets, bspline_config):
    """
    Args:
        coefficient_sets (dict): map of pair tuple to coefficient vector.
        bspline_config (bspline.BSplineConfig)

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