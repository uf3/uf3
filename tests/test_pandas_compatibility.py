"""Regression coverage for pandas-compatible parsing and database export."""
from pathlib import Path

import ase
import ase.db
import numpy as np
import pandas as pd

from uf3.data import io


def test_tsv_text_mixed_whitespace_and_sorted_ids():
    text = "id\tx   y\tz\n3\t1.25  -2.5\t3e-2\n1  -4\t5.5   6\n"

    actual = io.df_from_tsv_text(text)

    expected = pd.DataFrame(
        {"x": [-4.0, 1.25], "y": [5.5, -2.5], "z": [6.0, 0.03]},
        index=pd.Index([1, 3], name="id"),
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_lammps_log_preserves_values_across_timestep_reset():
    filename = Path(__file__).parent / "data" / "lammps" / "log.lammps"

    actual = io.parse_lammps_log(str(filename))

    assert list(actual.columns) == ["Step", "Temp", "PotEng", "Press", "Volume"]
    expected = np.array([
        [0, 600, -477.7349, -91919.895, 2108.9706],
        [1000, 390.48843, -459.76247, 356249.42, 1850.1756],
        [2000, 339.93182, -342.43985, 826614.06, 1576.0063],
        [3000, 280.01225, -418.58648, -247100.33, 2399.8398],
        [0, 8000, -418.58648, -190694.78, 2399.8398],
        [1000, 3707.642, -384.46704, -34852.696, 2267.7826],
        [2000, 4157.472, -381.45948, -11543.999, 2246.7761],
        [3000, 4024.4715, -391.86736, 21132.479, 2188.4333],
    ])
    np.testing.assert_allclose(actual.to_numpy(), expected, rtol=0, atol=1e-8)


def test_cache_data_preserves_structures_and_labels(tmp_path):
    names = ["frame_a", "frame_b"]
    symbols = ["W2", "W"]
    positions = [
        [[0.1, 0.2, 0.3], [2.8, 0.2, 0.4]],
        [[1.2, 2.3, 3.4]],
    ]
    energies = [-3.75, 0.125]
    forces = [
        np.array([[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]]),
        np.array([[-0.4, 0.5, -0.6]]),
    ]
    records = []
    for symbol, coords, energy, force in zip(symbols, positions, energies, forces):
        atoms = ase.Atoms(symbol, positions=coords, cell=[6, 7, 8],
                          pbc=[True, False, True])
        atoms.info["energy"] = energy
        record = {"geometry": atoms, "energy": energy, "size": len(atoms)}
        for axis, key in enumerate(("fx", "fy", "fz")):
            atoms.new_array(key, force[:, axis])
            record[key] = force[:, axis]
        records.append(record)
    frame = pd.DataFrame(records, index=names)
    filename = tmp_path / "structures.db"

    io.cache_data(frame, str(filename), serial=True)

    with ase.db.connect(str(filename), serial=True) as database:
        rows = list(database.select())
    assert len(rows) == len(records)
    for index, row in enumerate(rows):
        loaded = row.toatoms()
        assert row.row_name == names[index]
        assert loaded.get_chemical_formula() == symbols[index]
        np.testing.assert_allclose(loaded.positions, positions[index])
        np.testing.assert_allclose(loaded.cell.array, np.diag([6, 7, 8]))
        np.testing.assert_array_equal(loaded.pbc, [True, False, True])
        assert loaded.get_potential_energy() == energies[index]
        np.testing.assert_allclose(loaded.get_forces(), forces[index])
