import numpy as np
import ase
from scipy import linalg


def get_supercell(geometry, r_cut=10, sort_indices=False):
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
    supercell_factors = get_supercell_factors(cell, r_cut)
    radius_indices = [np.arange(n + 1) for n in supercell_factors]
    diameter_indices = [np.repeat(v, 2)[1:] for v in radius_indices]
    for i in range(3):
        diameter_indices[i][::2] *= -1

    a_indices, b_indices, c_indices = diameter_indices
    if sort_indices is False:
        sup_z = []
        sup_positions = []
        for a_idx in a_indices:
            for b_idx in b_indices:
                for c_idx in c_indices:
                    img_offset = np.dot([a_idx, b_idx, c_idx], cell)
                    img_positions = positions + img_offset
                    sup_positions.extend(img_positions.tolist())
                    sup_z.extend(z)
    else:  # sort images by distance to origin.
        a_grid, b_grid, c_grid = sort_image_indices(a_indices,
                                                    b_indices,
                                                    c_indices,
                                                    cell)
        n_images = len(a_grid)
        sup_z = np.tile(z, n_images)
        sup_positions = []
        for a_idx, b_idx, c_idx in zip(a_grid, b_grid, c_grid):
            img_offset = np.dot([a_idx, b_idx, c_idx], cell)
            img_positions = positions + img_offset
            sup_positions.extend(img_positions.tolist())
    supercell = ase.Atoms(sup_z, positions=sup_positions)
    return supercell


def get_supercell_factors(cell, r_cut):
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
    if np.any(np.max(cell, axis=1) == 0):
        raise ValueError("Unit cell has 0-length lattice vector(s).")
    normal_vectors = [np.cross(b, c),
                      np.cross(a, c),
                      np.cross(a, b)]
    projected_vectors = [n * np.dot(v, n) / np.dot(n, n)
                         for v, n in zip([a, b, c],
                                         normal_vectors)]
    supercell_factors = [r_cut / linalg.norm(p) for p in projected_vectors]
    supercell_factors = np.ceil(supercell_factors)
    return supercell_factors


def sort_image_indices(a_indices,
                       b_indices,
                       c_indices,
                       cell):
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


def energy_from_force_displacement(geom, energy, forces, d=0.01,
                                   n=None, random=True):
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
