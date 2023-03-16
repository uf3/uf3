"""
This module provides functions for generating supercells for
machine learning while avoiding the need for minimum-image convention.
"""

from typing import List, Collection, Tuple, Union
import warnings
import numpy as np
from scipy import linalg
import ase
from ase import cell as ase_cell


def get_supercell(geometry: ase.Atoms,
                  r_cut: float = 10,
                  sort_indices: bool = False
                  ) -> ase.Atoms:
    """
    Generate supercell, centered on original unit cell, with sufficient
    number of images along each lattice vector direction to ensure that
    atoms within the unit cell may interact with neighbors (in periodic
    images) at distances of r_cut or more.

    Args:
        geometry (ase.Atoms): configuration with periodic boundary conditions.
        r_cut (float): radial cutoff, default = 10.
        sort_indices (bool): sort images by distance to origin.

    Returns:
        supercell: ase.Atoms with maximum distances.
    """
    positions = geometry.get_positions()
    z = list(geometry.get_atomic_numbers())

    cell = geometry.get_cell()
    pbc = geometry.get_pbc()
    periodic_indices = generate_periodic_image_indices(cell, r_cut)
    # low-dimensional support
    for dim in range(len(periodic_indices)):
        if not pbc[dim]:
            periodic_indices[dim] = periodic_indices[dim][:1]
    # optionally sort images by distance to origin.
    periodic_grid = sort_image_indices(*periodic_indices,
                                       cell,
                                       sort=sort_indices)
    sup_positions, sup_z = tile_periodic_images(z,
                                                positions,
                                                cell,
                                                *periodic_grid)
    supercell = ase.Atoms(sup_z, positions=sup_positions)
    return supercell


def get_supercell_factors(cell: Union[ase_cell.Cell, np.ndarray],
                          r_cut: float = 10
                          ) -> np.ndarray:
    """
    Identify minimum number of replicas along each lattice vector direction
    to ensure that atoms within the unit cell may interact with neighbors
    (in periodic images) at distances of r_cut or more.

    Args:
        cell (ase.cell, np.ndarray): 3x3 array of lattice vectors.
        r_cut (float): cutoff radius.

    Returns:
        supercell_factors: minimum number of images per direction (radius).
    """
    a, b, c = cell
    if np.all(cell == 0):
        return [1, 1, 1]
    elif np.any(np.linalg.norm(cell, 2, axis=1) == 0):
        warnings.warn("Unit cell has 0-length lattice vector(s).")
        return [1, 1, 1]
    normal_vectors = [np.cross(b, c),
                      np.cross(a, c),
                      np.cross(a, b)]
    projected_vectors = [n * np.dot(v, n) / np.dot(n, n)
                         for v, n in zip([a, b, c],
                                         normal_vectors)]
    supercell_factors = [r_cut / linalg.norm(p) for p in projected_vectors]
    supercell_factors = np.ceil(supercell_factors)
    return supercell_factors


def sort_image_indices(a_indices: np.ndarray,
                       b_indices: np.ndarray,
                       c_indices: np.ndarray,
                       cell: np.ndarray,
                       sort: bool = True,
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort image indices based on distance to origin.

    Args:
        a_indices (np.ndarray): flattened coordinate array of image indices
            in a direction.
        b_indices (np.ndarray): flattened coordinate array of image indices
            in b direction.
        c_indices (np.ndarray): flattened coordinate array of image indices
            in c direction.
        cell (np.ndarray): 3x3 array of a, b, c lattice vectors.
        sort (bool): whether to perform sorting.

    Returns:
        Sorted, flattened coordinate arrays of image indices in each direction.
    """
    a_grid, b_grid, c_grid = np.meshgrid(a_indices,
                                         b_indices,
                                         c_indices,
                                         copy=False)
    a_grid = a_grid.flatten()
    b_grid = b_grid.flatten()
    c_grid = c_grid.flatten()

    if not sort:
        return a_grid, b_grid, c_grid

    img_centroids = [np.dot([a_idx, b_idx, c_idx], cell)
                     for a_idx, b_idx, c_idx
                     in zip(a_grid, b_grid, c_grid)]
    img_centroids = np.array(img_centroids)
    centroid_distances = linalg.norm(img_centroids, axis=1)
    centroid_sort = np.argsort(centroid_distances)
    a_grid = a_grid[centroid_sort]
    b_grid = b_grid[centroid_sort]
    c_grid = c_grid[centroid_sort]
    return a_grid, b_grid, c_grid


def generate_periodic_image_indices(cell, r_cut):
    supercell_factors = get_supercell_factors(cell, r_cut)
    radius_indices = [np.arange(n + 1) for n in supercell_factors]
    diameter_indices = [np.repeat(v, 2)[1:] for v in radius_indices]
    for i in range(3):
        diameter_indices[i][::2] *= -1
    a_indices, b_indices, c_indices = diameter_indices
    return [a_indices, b_indices, c_indices]


def tile_periodic_images(z, positions, cell, a_grid, b_grid, c_grid):
    n_images = len(a_grid)
    sup_z = np.tile(z, n_images)
    sup_positions = []
    for a_idx, b_idx, c_idx in zip(a_grid, b_grid, c_grid):
        img_offset = np.dot([a_idx, b_idx, c_idx], cell)
        img_positions = positions + img_offset
        sup_positions.extend(img_positions.tolist())
    return sup_positions, sup_z


def generate_displacements_from_forces(geom: ase.Atoms,
                                       energy: float,
                                       forces: Collection[Collection[float]],
                                       d: float = 0.01,
                                       n: int = None,
                                       random: bool = True
                                       ) -> Tuple[List[ase.Atoms], List]:
    """
    WIP implementation of data augmentation as introduced in
    https://doi.org/10.1038/s41524-020-0323-8
    """
    n_atoms = len(geom)
    positions = geom.get_positions()
    if random:  # n random displacements
        n = n or 25  # default
        displacements = [d * (np.random.rand(n_atoms, 3) * 2 - 1)
                         for _ in range(n)]  # small displacements
    else:  # 3N displacements
        n = n_atoms
        displacements = []
        for direction in [0, 1, 2]:  # cartesian directions
            d = np.ones(n) * d * np.sign(forces[:, direction])
            for atom_idx, position in enumerate(positions):  # atoms
                displacement = np.zeros_like(positions)
                displacement[atom_idx, direction] += d[atom_idx]
                displacements.append(displacement)
    snapshots = []
    energies = []
    for displacement in displacements:
        snapshot = geom.copy()
        snapshot.translate(displacement)
        snapshots.append(snapshot)
        de = -np.sum(np.multiply(forces, displacement))  # F = -dE/dR
        energies.append(energy + de)
    return snapshots, energies
