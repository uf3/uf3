import pytest
from uf3.data.geometry import *
from uf3.representation.distances import *
import itertools


@pytest.fixture()
def simple_molecule():
    geom = ase.Atoms('Ar3',
                     positions=[[0, 0, 0], [3, 0.0, 0.0], [0, 4.0, 0]],
                     pbc=False,
                     cell=None)
    yield geom


@pytest.fixture()
def simple_unary():
    geom = ase.Atoms('Au2',
                     positions=[[0, 0, 0], [0.5, 0.3, 0.2]],
                     pbc=True,
                     cell=[[2, 0, 0], [3, 1.5, 0], [0.5, 0, 2.5]])
    yield geom


@pytest.fixture()
def simple_binary():
    geom = ase.Atoms('NeXe',
                     positions=[[0, 0, 0], [0.5, 0.3, 0.2]],
                     pbc=True,
                     cell=[[2, 0, 0], [3, 1.5, 0], [0.5, 0, 2.5]])
    yield geom


class TestMolecule:
    def test_distances(self, simple_molecule):
        geom = simple_molecule
        # parameters
        element_list = ['Ar']
        cwr = itertools.combinations_with_replacement(element_list, 2)
        pair_tuples = sorted(cwr)
        r_min_map = {('Ar', 'Ar'): 0.5}
        r_max_map = {('Ar', 'Ar'): 6.0}
        # compute
        distances = distances_by_interaction(geom,
                                             pair_tuples,
                                             r_min_map,
                                             r_max_map,
                                             atomic=False)
        d_aa = distances[('Ar', 'Ar')]
        assert len(d_aa) == 6
        assert np.allclose(np.sort(d_aa), [3, 3, 4, 4, 5, 5])

    def test_distance_derivatives(self, simple_molecule):
        geom = simple_molecule
        supercell = simple_molecule
        # parameters
        element_list = ['Ar']
        cwr = itertools.combinations_with_replacement(element_list, 2)
        pair_tuples = sorted(cwr)
        r_min_map = {('Ar', 'Ar'): 0.5, }
        r_max_map = {('Ar', 'Ar'): 6.0, }
        # compute
        r_cut = 6.0
        distances, derivatives = derivatives_by_interaction(geom,
                                                            pair_tuples,
                                                            r_cut,
                                                            r_min_map,
                                                            r_max_map,
                                                            supercell,)
        d_aa = distances[('Ar', 'Ar')]
        assert len(d_aa) == 6
        dr_aa = derivatives[('Ar', 'Ar')]
        assert dr_aa.shape == (3, 3, 6)



class TestUnary:
    def test_distances(self, simple_unary):
        geom = simple_unary
        supercell = get_supercell(geom, r_cut=4)
        # parameters
        element_list = ['Au']
        cwr = itertools.combinations_with_replacement(element_list, 2)
        pair_tuples = sorted(cwr)
        r_min_map = {('Au', 'Au'): 0.5, }
        r_max_map = {('Au', 'Au'): 3.0, }
        # compute
        distances = distances_by_interaction(geom,
                                             pair_tuples,
                                             r_min_map,
                                             r_max_map,
                                             supercell=supercell,
                                             atomic=False)
        d_aa = distances[('Au', 'Au')]
        assert len(d_aa) == 58
        assert np.min(d_aa) >= r_min_map[('Au', 'Au')]
        assert np.max(d_aa) <= r_max_map[('Au', 'Au')]

    def test_distance_derivatives(self, simple_unary):
        geom = simple_unary
        supercell = get_supercell(geom, r_cut=4)
        # parameters
        element_list = ['Au']
        cwr = itertools.combinations_with_replacement(element_list, 2)
        pair_tuples = sorted(cwr)
        r_min_map = {('Au', 'Au'): 0.5, }
        r_max_map = {('Au', 'Au'): 3.0, }
        # compute
        r_cut = 3.0
        distances, derivatives = derivatives_by_interaction(geom,
                                                            pair_tuples,
                                                            r_cut,
                                                            r_min_map,
                                                            r_max_map,
                                                            supercell,)
        d_aa = distances[('Au', 'Au')]
        assert len(d_aa) == 542
        dr_aa = derivatives[('Au', 'Au')]
        assert dr_aa.shape == (2, 3, 542)


class TestBinary:
    def test_distances(self, simple_binary):
        geom = simple_binary
        supercell = get_supercell(geom, r_cut=4)
        # parameters
        element_list = ['Ne', 'Xe']
        cwr = itertools.combinations_with_replacement(element_list, 2)
        pair_tuples = sorted(cwr)
        r_min_map = {('Ne', 'Ne'): 0.5,
                     ('Ne', 'Xe'): 0.6,
                     ('Xe', 'Xe'): 0.7, }
        r_max_map = {('Ne', 'Ne'): 3.0,
                     ('Ne', 'Xe'): 4.0,
                     ('Xe', 'Xe'): 5.0, }
        # compute
        distances = distances_by_interaction(geom,
                                             pair_tuples,
                                             r_min_map,
                                             r_max_map,
                                             supercell=supercell,
                                             atomic=False)
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
        geom = simple_binary
        supercell = get_supercell(geom, r_cut=4)
        # parameters
        element_list = ['Ne', 'Xe']
        cwr = itertools.combinations_with_replacement(element_list, 2)
        pair_tuples = sorted(cwr)
        r_min_map = {('Ne', 'Ne'): 0.5,
                     ('Ne', 'Xe'): 0.6,
                     ('Xe', 'Xe'): 0.7, }
        r_max_map = {('Ne', 'Ne'): 3.0,
                     ('Ne', 'Xe'): 4.0,
                     ('Xe', 'Xe'): 5.0, }
        # compute
        r_cut = 5.0
        distances, derivatives = derivatives_by_interaction(geom,
                                                            pair_tuples,
                                                            r_cut,
                                                            r_min_map,
                                                            r_max_map,
                                                            supercell)
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
