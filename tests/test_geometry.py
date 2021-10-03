import pytest
from uf3.data.geometry import *


@pytest.fixture()
def simple_structure():
    geometry = ase.Atoms('Au2',
                         positions=[[0, 0, 0], [0.5, 0.3, 0.2]],
                         pbc=True,
                         cell=[[2, 0, 0], [3, 1.5, 0], [0.5, 0, 2.5]])
    yield geometry


class TestGeometry:
    def test_supercell_factors(self, simple_structure):
        cell = simple_structure.get_cell()
        supercell_factors = get_supercell_factors(cell, 1e-6).tolist()
        target = [1., 1., 1.]
        assert supercell_factors == target
        supercell_factors = get_supercell_factors(cell, 2).tolist()
        target = [3., 2., 1.]
        assert supercell_factors == target

    def test_supercell(self, simple_structure):
        supercell = get_supercell(simple_structure, r_cut=1e-6)
        n_atoms = len(supercell)
        assert n_atoms == 54
        supercell = get_supercell(simple_structure, r_cut=2)
        n_atoms = len(supercell)
        assert n_atoms == 210

    def test_sort_image_indices(self):
        a_indices = [0, 0, 1, 1],
        b_indices = [2, -2, 0, 1],
        c_indices = [5, 7, 1, 5],
        cell = [[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]]
        a_grid, b_grid, c_grid = sort_image_indices(a_indices,
                                                    b_indices,
                                                    c_indices,
                                                    cell)
        centroids = []
        for a_idx, b_idx, c_idx in zip(a_grid, b_grid, c_grid):
            centroids.append(np.dot([a_idx, b_idx, c_idx], cell))
        distances = linalg.norm(centroids, axis=1)
        gradient = np.gradient(distances)
        assert np.min(gradient) >= 0

    def test_energy_force_augment(self, simple_structure):
        energy = 1
        forces = np.array([[0.1, 0.2, 0.3],
                           [-0.11, -0.22, -0.33]])
        snapshots, energies = generate_displacements_from_forces(simple_structure,
                                                                 energy,
                                                                 forces,
                                                                 d=0.01,
                                                                 random=False)
        assert len(snapshots) == 6
        assert len(energies) == 6
        snapshots, energies = generate_displacements_from_forces(simple_structure,
                                                                 energy,
                                                                 forces,
                                                                 d=0.01,
                                                                 n=7)
        assert len(snapshots) == 7
        assert len(energies) == 7