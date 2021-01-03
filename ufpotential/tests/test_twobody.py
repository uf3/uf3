import pytest
from ufpotential.data.geometry import *
from ufpotential.data.two_body import *

from itertools import combinations_with_replacement as cwr


@pytest.fixture()
def simple_unary():
    geometry = ase.Atoms('Au2',
                         positions=[[0, 0, 0], [0.5, 0.3, 0.2]],
                         pbc=True,
                         cell=[[2, 0, 0], [3, 1.5, 0], [0.5, 0, 2.5]])
    yield geometry


@pytest.fixture()
def simple_binary():
    geometry = Atoms('NeXe',
                     positions=[[0, 0, 0], [0.5, 0.3, 0.2]],
                     pbc=True,
                     cell=[[2, 0, 0], [3, 1.5, 0], [0.5, 0, 2.5]])
    yield geometry


class TestUnary:
    def test_distances(self, simple_unary):
        geometry = simple_unary
        supercell = get_supercell(geometry, r_cut=4)
        # parameters
        element_list = ['Au']
        interactions_map = {2: sorted(cwr(element_list, 2))}
        r_min_map = {('Au', 'Au'): 0.5, }
        r_max_map = {('Au', 'Au'): 3.0, }
        # compute
        distances = distances_by_interaction(geometry,
                                             interactions_map,
                                             r_min_map,
                                             r_max_map,
                                             supercell=supercell,
                                             average=True)
        d_aa = distances[('Au', 'Au')]
        assert len(d_aa) == 58
        assert np.min(d_aa) >= r_min_map[('Au', 'Au')]
        assert np.max(d_aa) <= r_max_map[('Au', 'Au')]

    def test_distance_derivatives(self, simple_unary):
        geometry = simple_unary
        supercell = get_supercell(geometry, r_cut=4)
        # parameters
        element_list = ['Au']
        interactions_map = {2: sorted(cwr(element_list, 2))}
        r_min_map = {('Au', 'Au'): 0.5, }
        r_max_map = {('Au', 'Au'): 3.0, }
        # compute
        distances, derivatives = derivatives_by_interaction(geometry,
                                                            supercell,
                                                            interactions_map,
                                                            r_min_map,
                                                            r_max_map)
        d_aa = distances[('Au', 'Au')]
        assert len(d_aa) == 542
        dr_aa = derivatives[('Au', 'Au')]
        assert dr_aa.shape == (2, 3, 542)


class TestBinary:
    def test_distances(self, simple_binary):
        geometry = simple_binary
        supercell = get_supercell(geometry, r_cut=4)
        # parameters
        element_list = ['Ne', 'Xe']
        interactions_map = {2: sorted(cwr(element_list, 2))}
        r_min_map = {('Ne', 'Ne'): 0.5,
                     ('Ne', 'Xe'): 0.6,
                     ('Xe', 'Xe'): 0.7, }
        r_max_map = {('Ne', 'Ne'): 3.0,
                     ('Ne', 'Xe'): 4.0,
                     ('Xe', 'Xe'): 5.0, }
        # compute
        distances = distances_by_interaction(geometry,
                                             interactions_map,
                                             r_min_map,
                                             r_max_map,
                                             supercell=supercell,
                                             average=True)
        d_aa = distances[('Ne', 'Ne')]
        d_ab = distances[('Ne', 'Xe')]
        d_bb = distances[('Xe', 'Xe')]
        assert len(d_aa) == 14
        assert len(d_ab) == 74
        assert len(d_bb) == 58
        assert np.min(d_aa) >= r_min_map[('Ne', 'Ne')]
        assert np.max(d_aa) <= r_max_map[('Ne', 'Ne')]
        assert np.min(d_ab) >= r_min_map[('Ne', 'Xe')]
        assert np.max(d_ab) <= r_max_map[('Ne', 'Xe')]
        assert np.min(d_bb) >= r_min_map[('Xe', 'Xe')]
        assert np.max(d_bb) <= r_max_map[('Xe', 'Xe')]

    def test_distance_derivatives(self, simple_binary):
        geometry = simple_binary
        supercell = get_supercell(geometry, r_cut=4)
        # parameters
        element_list = ['Ne', 'Xe']
        interactions_map = {2: sorted(cwr(element_list, 2))}
        r_min_map = {('Ne', 'Ne'): 0.5,
                     ('Ne', 'Xe'): 0.6,
                     ('Xe', 'Xe'): 0.7, }
        r_max_map = {('Ne', 'Ne'): 3.0,
                     ('Ne', 'Xe'): 4.0,
                     ('Xe', 'Xe'): 5.0, }
        # compute
        distances, derivatives = derivatives_by_interaction(geometry,
                                                            supercell,
                                                            interactions_map,
                                                            r_min_map,
                                                            r_max_map)
        d_aa = distances[('Ne', 'Ne')]
        d_ab = distances[('Ne', 'Xe')]
        d_bb = distances[('Xe', 'Xe')]
        assert len(d_aa) == 660
        assert len(d_ab) == 3102
        assert len(d_bb) == 2142
        dr_aa = derivatives[('Ne', 'Ne')]
        dr_ab = derivatives[('Ne', 'Xe')]
        dr_bb = derivatives[('Xe', 'Xe')]
        assert dr_aa.shape == (2, 3, 660)
        assert dr_ab.shape == (2, 3, 3102)
        assert dr_bb.shape == (2, 3, 2142)


class TestUnaryLegacy:
    """Legacy functions"""

    def test_distances(self, simple_unary):
        supercell = get_supercell(simple_unary, r_cut=2)
        distances = distances_from_geometry(simple_unary,
                                            supercell,
                                            r_min=0.5,
                                            r_max=2)
        assert len(distances) == 18
        assert np.min(distances) >= 0.5
        assert np.max(distances) <= 2

    def test_distance_derivatives(self, simple_unary):
        supercell = get_supercell(simple_unary, r_cut=1e-6)
        distances, derivatives = get_distance_derivatives(simple_unary,
                                                          supercell,
                                                          r_min=1e-6,
                                                          r_max=2)
        assert len(distances.flatten()) == 50
        assert derivatives.shape == (2, 3, 50)
