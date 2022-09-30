from typing import Dict, Tuple, Union, Any
import warnings
import numpy as np
from scipy import spatial
import ase
from scipy import signal
from scipy import optimize as sp_opt
from sklearn import linear_model as sk_linear
from uf3.data import geometry
from uf3.data import composition
from uf3.util import parallel


def get_uniform_normalization(bins, n_atoms, volume):
    """Reference ideal-gas number density"""
    edges_inner = bins[:-1]
    edges_outer = bins[1:]
    v_ideal = volume / n_atoms
    v_outer = 4 / 3 * np.pi * (edges_outer ** 3)
    v_inner = 4 / 3 * np.pi * (edges_inner ** 3)
    v_shell = (v_outer - v_inner) / v_ideal * n_atoms
    return v_shell


def apply_binning(dist_ref, bins):
    return {k: np.histogram(dists, bins)[0]
            for k, dists in dist_ref.items()}


def score_coverage(x, histogram, reference, weight=10):
    """loss function for identifying the maximum reference density
     under the curve"""
    histogram = np.array(histogram)
    lbound = np.where(np.nonzero(histogram))[0][0]
    reference = reference * x
    delta = histogram - reference
    positive = reference[delta >= 0][lbound:]
    negative = delta[delta < 0][lbound:]
    score = np.sum(positive) + np.sum(negative * weight)
    return -score


def compute_coverage(x, histogram, reference):
    """Area under the maximal reference curve corresponding to the maximum
    uniform-density subset spanning the training data"""
    delta = histogram - reference * x
    delta[delta < 0] = 0
    coverage = histogram - delta
    return np.sum(coverage)


def suggest_cutoffs(lower_bound, valley_list, bond_length):
    print(f"    Smallest observed: {lower_bound:.2f} angstroms")
    valley_slice = valley_list[valley_list >= bond_length]
    print("    Suggested Cutoffs:", valley_slice)


class DataAnalyzer:
    def __init__(self,
                 chemical_system: composition.ChemicalSystem,
                 r_cut: float = 12.0,
                 rattle: float = 0.0,
                 bins: Union[int, float] = 0.01,
                 min_peak_width: float = 0.2,
                 progress: Any = "bar"):
        """
        Args:
            chemical_system (uf3.composition.ChemicalSystem)
            r_cut (float): cutoff distance in angstroms.
            rattle: amplitude for random perturbation of atoms. Useful for
                broadening peaks in RDF.
            bins (int, float): specify bin resolution/frequency for histogram.
                float: sets spatial resolution per bin (angstroms)
                int: sets total number of bins
            min_peak_width (float): minimum peak with in angstroms for
                peak-finding algorithm.
            progress: style of progress bar.
        """
        self.chemical_system = chemical_system
        self.r_cut = r_cut
        self.rattle = rattle
        self.min_peak_width = min_peak_width
        self.progress = progress
        # initialize composition-volume quantities
        self.element_set = chemical_system.numbers
        self.element_names = chemical_system.element_list
        self.n_elements = len(self.element_set)
        self.pair_tuples = chemical_system.interactions_map[2]
        # initialize histogram quantities
        if isinstance(bins, int):
            self.n_bins = bins
        else:
            self.n_bins = int(np.ceil(r_cut / bins))
        self.bin_edges = np.linspace(0, r_cut, self.n_bins + 1)
        self.bin_width = np.mean(self.bin_edges[1::2]
                                 - self.bin_edges[:-1:2])
        self.bin_centers = 0.5 * np.add(self.bin_edges[:-1],
                                        self.bin_edges[1:])
        self.bin_span = int(np.ceil(min_peak_width / self.bin_width))
        self.clear()

        self.outliers = {}

    def clear(self):
        """Reset accumulated pair and volume data."""
        self.histogram_values = {}
        self.pairs_acc = {}  # number of pairs per hash
        self.totals_acc = 0  # total pairs

        self.sizes = []  # number of atoms per sample
        self.volumes = []  # volume per sample
        self.compositions = []  # composition per sample

        self.lower_bounds = {}
        self.peaks = {}
        self.valleys = {}
        self.volume_ref = {}
        self.radii_ref = {}
        self.density_ref = {}
        self.normalized_values = {}

    def get_distances(self, geom, r_min=None, r_max=None, rattle=0.0):
        """

        Args:
            geom: ase.Atoms
            r_min:
            r_max:
            rattle: amplitude for random perturbation of atoms. Useful for
                broadening peaks in RDF.

        Returns:
            dists (np.ndarray): flattened array of pair distances
            hashes (np.hdarray): flattened array of pair hashes
        """
        n_atoms = len(geom)
        if r_min is None:
            r_min = 0.0
        if r_max is None:
            r_max = self.r_cut
        if any(geom.pbc):
            supercell = geometry.get_supercell(geom, r_cut=r_max)
        else:
            supercell = geom
        if rattle > 0:
            supercell.rattle(rattle)
        dist_matr = spatial.distance.cdist(geom.get_positions(),
                                           supercell.get_positions())
        cut_mask = (dist_matr > r_min) & (dist_matr <= r_max)
        dists = dist_matr[cut_mask]
        pair_idx = np.where(cut_mask)
        species_set = (geom.get_atomic_numbers(),
                       supercell.get_atomic_numbers())
        symbols_set = (geom.get_chemical_symbols(),
                       supercell.get_chemical_symbols())
        hashes = composition.get_pair_hashes(species_set, symbols_set, pair_idx)
        return dists, hashes

    def process_geometry(self,
                         geom: ase.Atoms,
                         r_min: float = None,
                         r_max: float = None,
                         rattle: float = None,
                         ):
        """
        Construct histogram of distances per pair interaction across
            list of geometries. Useful for optimizing the lower- and upper-
            bounds of knot sequences.

        """
        if r_min is None:
            r_min = 0.0
        if r_max is None:
            r_max = self.r_cut
        if rattle is None:
            rattle = self.rattle
        numbers = geom.get_atomic_numbers()
        comp = [np.count_nonzero(numbers == el)
                for el in self.element_set]
        if any([n not in self.element_set for n in numbers]):
            warnings.warn(f"Invalid element detected: {numbers}")
        self.sizes.append(len(geom))
        self.volumes.append(geom.get_volume())
        self.compositions.append(comp)
        dists, hashes = self.get_distances(geom,
                                           r_min,
                                           r_max,
                                           rattle=rattle)
        dist_ref = composition.hash_gather(dists, hashes)
        self.update_histograms(dist_ref)

    def update_histograms(self, dist_ref):
        histograms = apply_binning(dist_ref, self.bin_edges)
        for key in histograms:
            n_pairs = np.sum(histograms[key])
            if key not in self.histogram_values:
                self.histogram_values[key] = np.zeros(self.n_bins)
                self.pairs_acc[key] = 0
            self.histogram_values[key] += histograms[key]
            self.pairs_acc[key] += n_pairs
            self.totals_acc += n_pairs

    def load_entries(self, geometries):
        n_entries = len(geometries)
        iterable = parallel.progress_iter(enumerate(geometries),
                                          style=self.progress,
                                          total=n_entries)
        for j, geom in iterable:
            self.process_geometry(geom)

    def normalize_pair_histogram(self, pair, n_atoms, volume):
        norm = get_uniform_normalization(self.bin_edges,
                                         n_atoms,
                                         volume)
        weight = self.pairs_acc[pair] / self.totals_acc
        rdf = self.histogram_values[pair] / norm / weight
        return norm, rdf

    def analyze(self,
                smooth: bool = True,
                filter_width: int = 9,
                filter_degree: int = 3,
                ) -> Dict:
        """
        Construct histogram of distances per pair interaction across
            list of geometries. Useful for optimizing the lower- and upper-
            bounds of knot sequences.

        Args:
            smooth (bool):
            filter_width (int):
            filter_degree (int):

        """
        reference = {}
        rdfs = {}
        coverages = {}
        factors = {}
        symbol_map = {k: composition.hash_to_symbols(k)
                      for k in self.histogram_values.keys()}

        hash_check = [isinstance(k, int) for k in self.pairs_acc.keys()]
        if any(hash_check):
            self.pairs_acc = {symbol_map[k]: v for k, v
                               in self.pairs_acc.items()}
            self.histogram_values = {symbol_map[k]: v for k, v
                                     in self.histogram_values.items()}

        # estimate average bond lengths
        atomic_volumes, volume_soln = self.fit_element_data()
        bond_ref = {}
        for pair in self.pair_tuples:
            bond_ref[pair] = (np.mean([atomic_volumes[el] for el in pair])
                              / (4 / 3 * np.pi)) ** (1 / 3) * 2
        # process pair data
        n_atoms = np.sum(self.sizes)
        volume = np.sum(self.volumes)
        for pair in self.pair_tuples:
            hist = self.histogram_values[pair]
            if np.sum(self.histogram_values[pair]) == 0:
                warnings.warn(f"No observed {pair} pairs.")
                continue

            norm, rdf = self.normalize_pair_histogram(pair, n_atoms, volume)
            rdfs[pair] = rdf
            reference[pair] = norm
            coverage_scale = sp_opt.minimize(score_coverage,
                                             1,
                                             args=(hist, norm, 10),
                                             method="Nelder-Mead").x
            coverage = compute_coverage(coverage_scale, hist, norm)
            coverages[pair] = coverage
            self.find_pair_distribution_peaks(pair,
                                              smooth=smooth,
                                              filter_width=filter_width,
                                              filter_degree=filter_degree)
            suggest_cutoffs(self.lower_bounds[pair],
                            self.valleys[pair],
                            bond_ref[pair])
        analysis = dict(histograms=self.histogram_values,
                        bin_edges=self.bin_edges,
                        reference=reference,
                        rdfs=rdfs,
                        coverage=coverages,
                        factors=factors,
                        lower_bounds=self.lower_bounds,
                        peaks=self.peaks,
                        valleys=self.valleys,
                        atomic_volumes=atomic_volumes,
                        )
        return analysis

    def fit_element_data(self):
        """
        Fit observed cell volume per atom.

        TO DO: contextualize with rock salt (FCC), cesium-chloride (BCC),
        zinc blende (diamond) examples, packing fractions,
        and nearest neighbor distances.
        """
        x_ridge = np.ones((self.n_elements, self.n_elements)) * 1e-6
        y_ridge = np.zeros(self.n_elements)
        x_volume = np.concatenate([self.compositions, x_ridge])
        y_volume = np.concatenate([self.volumes, y_ridge])
        volume_regr = sk_linear.HuberRegressor(fit_intercept=False)
        volume_regr.fit(x_volume, y_volume)
        volume_soln = volume_regr.coef_
        atomic_volumes = {el: value for el, value
                          in zip(self.element_names, volume_soln)}
        return atomic_volumes, volume_soln

    def find_pair_distribution_peaks(self,
                                     pair: Tuple,
                                     smooth: bool = True,
                                     filter_width: int = 9,
                                     filter_degree: int = 3,
                                     ):
        histogram_values = self.histogram_values[pair]
        lower_idx = np.nonzero(histogram_values)[0][0]
        lower_bound = self.bin_edges[lower_idx]
        self.lower_bounds[pair] = lower_bound
        peak_idx, peak_list = find_peaks(self.bin_centers,
                                         histogram_values,
                                         smooth=smooth,
                                         filter_width=filter_width,
                                         filter_degree=filter_degree)
        valley_list = np.mean([peak_list[1:], peak_list[:-1]], axis=0)
        self.valleys[pair] = valley_list
        self.peaks[pair] = peak_list


def find_peaks(x,
               y,
               smooth=False,
               filter_width=9,
               filter_degree=3,
               ):
    if smooth:
        y = signal.savgol_filter(y, filter_width, filter_degree)
    peak_idx = signal.find_peaks(y)[0]
    peak_list = x[peak_idx]
    return peak_idx, peak_list


def find_closest_value(values, target):
    scores = np.abs(values - target)
    idx = np.argmin(scores)
    return idx, values[idx]
