"""Regression cases for the repository audit, including silent wrong results."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import copy
import importlib.util
import json

import ase
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
import pandas as pd
import pytest
import yaml

from uf3.data import composition, geometry, io
from uf3.forcefield import calculator
from uf3.regression import least_squares
from uf3.representation import bspline, process
from uf3.util import json_io, parallel, subsample, user_config


@pytest.fixture
def basis():
    return bspline.BSplineBasis(composition.ChemicalSystem(['W']))


@pytest.fixture
def model():
    path = Path(__file__).parent / 'data/precalculated_ref/model_unary.json'
    return least_squares.WeightedLinearModel.from_json(str(path))


def test_copy_preserves_calculator_labels_without_mutating_original():
    atoms = ase.Atoms('W2', positions=[[0, 0, 0], [2.8, 0, 0]])
    atoms.calc = SinglePointCalculator(atoms, energy=3., forces=np.ones((2, 3)))
    df = io.prepare_dataframe_from_lists([atoms])
    assert df.energy.iloc[0] == 3.
    np.testing.assert_array_equal(df.fx.iloc[0], np.ones(2))
    assert 'energy' not in atoms.info
    assert 'fx' not in atoms.arrays
    assert df.geometry.iloc[0] is not atoms


def test_explicit_labels_override_existing_arrays_and_custom_keys():
    atoms = ase.Atoms('W2')
    atoms.new_array('fx', np.zeros(2))
    df = io.prepare_dataframe_from_lists([atoms], energies=[4.],
        forces=[np.ones((2, 3))], atoms_key='atoms', energy_key='E')
    assert df.E.iloc[0] == df.atoms.iloc[0].info['E'] == 4.
    np.testing.assert_array_equal(df.atoms.iloc[0].arrays['fx'], np.ones(2))


def test_regularization_calls_are_independent(basis):
    expected = basis.get_regularization_matrix()
    ridge = {2: .25}
    curve = {2: .5}
    basis.get_regularization_matrix(ridge_map=ridge, curvature_map=curve,
                                    ridge_2b=100., curvature_2b=10.)
    np.testing.assert_array_equal(basis.get_regularization_matrix(), expected)
    assert ridge == {2: .25}
    assert curve == {2: .5}


@pytest.mark.parametrize('pbc,cell', [([True, True, False], [2., 2., 0.]),
                                    ([True, False, False], [2., 0., 0.])])
def test_low_dimensional_cell_vacuum_invariance(model, pbc, cell):
    atoms = ase.Atoms('W', cell=cell, pbc=pbc)
    atoms.calc = calculator.UFCalculator(model)
    expected = atoms.get_potential_energy()
    with_vacuum = [length if periodic else 10. for length, periodic in zip(cell, pbc)]
    atoms.set_cell(with_vacuum)
    assert atoms.get_potential_energy() == pytest.approx(expected, abs=1e-10)


def test_dependent_periodic_vectors_are_rejected():
    atoms = ase.Atoms('W', cell=[[2, 0, 0], [4, 0, 0], [0, 0, 0]],
                      pbc=[True, True, False])
    with pytest.raises(ValueError, match='independent'):
        geometry.get_supercell(atoms)


@pytest.mark.parametrize('weight', [0., .5, 1.])
def test_constant_force_targets_remain_finite(basis, weight):
    model = least_squares.WeightedLinearModel(basis, regularizer=np.eye(basis.n_feats))
    rng = np.random.default_rng(14)
    model.fit(rng.random((4, basis.n_feats)), np.arange(4.),
              rng.random((12, basis.n_feats)), np.zeros(12), weight=weight)
    assert np.all(np.isfinite(model.coefficients))


def test_energy_only_file_fit_matches_memory_fit(basis, tmp_path):
    atoms = [ase.Atoms('W2', positions=[[0, 0, 0], [d, 0, 0]])
             for d in np.linspace(1.5, 3., 5)]
    data = io.prepare_dataframe_from_lists(atoms, energies=np.arange(5.), prefix='e')
    features = process.BasisFeaturizer(basis, fit_forces=False).evaluate(data, progress=False)
    path = tmp_path / 'energies.h5'
    features.to_hdf(path, key='chunk0')
    model = least_squares.WeightedLinearModel(basis, regularizer=np.eye(basis.n_feats))
    model.fit_from_file(str(path), data.index, weight=1., progress=False)
    x, y, _, _ = least_squares.dataframe_to_tuples(features, n_elements=1)
    reference = least_squares.WeightedLinearModel(basis, regularizer=np.eye(basis.n_feats))
    reference.fit(x, y)
    np.testing.assert_allclose(model.coefficients, reference.coefficients)


def test_no_matching_training_rows_is_an_error(basis, tmp_path):
    path = tmp_path / 'features.h5'
    df = pd.DataFrame([[0.] * (basis.n_feats + 1)], columns=basis.get_column_names(),
                      index=pd.MultiIndex.from_tuples([('a', 'energy')]))
    df.to_hdf(path, key='chunk0')
    with pytest.raises(ValueError, match='samples'):
        least_squares.WeightedLinearModel(basis).fit_from_file(
            str(path), ['missing'], progress=False)


def test_invalid_training_values_are_rejected(basis):
    model = least_squares.WeightedLinearModel(basis)
    with pytest.raises(ValueError, match='finite'):
        model.fit(np.ones((2, basis.n_feats)), np.array([1., np.nan]))


def test_legacy_training_weights_handle_constant_and_missing_blocks():
    for labels in [[('a', 'energy'), ('b', 'energy')],
                   [('a', 'energy'), ('a', 'fx_0')]]:
        df = pd.DataFrame([[1., 2.], [1., 2.]],
                          index=pd.MultiIndex.from_tuples(labels))
        _, _, weights = process.dataframe_to_training_tuples(df)
        assert np.all(np.isfinite(weights))


def test_configuration_pipeline_and_defaults(tmp_path):
    path = tmp_path / 'settings.yaml'
    path.write_text('elements: [W]\nbasis:\n  r_max:\n    W-W: 5.0\n'
                    'features:\n  fit_forces: false\n  column_prefix: z\n')
    settings = user_config.read_config(path)
    before = copy.deepcopy(settings)
    handlers = user_config.generate_handlers(settings)
    assert {'data', 'chemical_system', 'basis', 'features', 'learning'} <= handlers.keys()
    assert not handlers['features'].fit_forces
    assert handlers['features'].prefix == 'z'
    assert settings == before
    assert handlers['basis'].degree == 2
    assert handlers['basis'].r_max_map[('W', 'W')] == 5.


def test_saved_model_handler(tmp_path, model):
    path = tmp_path / 'model.json'
    model.to_json(str(path))
    settings = {'elements': ['W'], 'degree': 3, 'basis': model.bspline_config.as_dict(),
                'model': {'model_path': str(path)}}
    restored = user_config.generate_handlers(settings)['model']
    np.testing.assert_allclose(restored.coefficients, model.coefficients)


def test_yaml_python_tags_are_rejected(tmp_path):
    path = tmp_path / 'unsafe.yaml'
    path.write_text('!!python/object/apply:builtins.eval ["{\'seed\': 37}"]')
    with pytest.raises(yaml.YAMLError):
        user_config.read_config(path)


@pytest.mark.parametrize('contents', ['', '[1, 2]', 'features: 42'])
def test_invalid_settings_are_reported(tmp_path, contents):
    path = tmp_path / 'settings.yaml'
    path.write_text(contents)
    if not contents:
        assert user_config.read_config(path)['degree'] == 2
    else:
        with pytest.raises(ValueError, match='mapping'):
            user_config.read_config(path)


def test_basis_config_allows_default_trims():
    basis = bspline.BSplineBasis.from_config({'element_list': ['W'], 'degree': 2})
    assert basis.leading_trim[2] == 0


def test_stress_and_force_consistent_energy(model):
    atoms = ase.Atoms('W2', positions=[[0, 0, 0], [2.8, .2, 0]], cell=[6, 6, 6], pbc=True)
    atoms.calc = calculator.UFCalculator(model)
    assert atoms.get_potential_energy(force_consistent=True) == atoms.get_potential_energy()
    assert atoms.get_stress().shape == (6,)
    assert np.all(np.isfinite(atoms.get_stress()))


@pytest.mark.parametrize('relax_cell', [False, True])
def test_relaxation_runs_and_improves_energy(model, relax_cell):
    atoms = ase.Atoms('W2', positions=[[0, 0, 0], [2.8, 0, 0]], cell=[6, 6, 6], pbc=True)
    calc = calculator.UFCalculator(model)
    start = calc.get_potential_energy(atoms)
    relaxed = calc.relax_fmax(atoms, relax_cell=relax_cell, steps=2)
    assert np.isfinite(relaxed.get_potential_energy())
    assert relaxed.get_potential_energy() <= start + 1e-8


def test_current_phonopy_integration(model):
    pytest.importorskip('phonopy')
    pytest.importorskip('seekpath')
    from uf3.forcefield.properties import phonon
    atoms = ase.Atoms('W', cell=[3., 3., 3.], pbc=True)
    constants, path, bands = phonon.compute_phonon_data(
        atoms, calculator.UFCalculator(model), n_super=1, resolution=2)
    assert np.all(np.isfinite(constants))
    assert path['path']
    assert len(bands['frequencies']) == len(path['path'])


def test_exporter_cli_and_scalar_trim_headers(tmp_path, model, capsys, monkeypatch):
    path = Path(__file__).parents[1] / 'lammps_plugin/scripts/generate_uf3_lammps_pots.py'
    spec = importlib.util.spec_from_file_location('uf3_exporter', path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    model_path = tmp_path / 'model.json'
    model.to_json(str(model_path))
    monkeypatch.setattr('sys.argv', [str(path), '-a', 'test', '-u', 'metal',
                                    '-m', str(model_path), '-d', str(tmp_path)])
    exporter.main()
    assert 'pair_style\tuf3 3\n' in capsys.readouterr().out
    lines = (tmp_path / 'W.uf3').read_text().splitlines()
    assert '2B W W 0 3 nk' in lines
    assert '3B W W W 0 3 nk' in lines


@pytest.mark.parametrize('data', [np.ones(4), np.array([[4., 3.], [2., 1.]]),
                               np.array([])])
def test_sampling_is_in_bounds_and_distinct(data):
    indices = subsample.farthest_point_sampling(data, max_samples=4)
    assert len(indices) == len(data)
    assert len(set(indices)) == len(indices)
    assert np.all((indices >= 0) & (indices < len(data)))


def test_sampling_separation():
    data = np.array([0., 0., 1., 2.])
    indices = subsample.farthest_point_sampling(data, min_diff=.5)
    assert len(indices) == 3


def test_weighted_scoring_nonsquare_matrix():
    model = least_squares.BasicLinearModel()
    model.coefficients = np.array([1., 2.])
    x = np.array([[1., 0.], [0., 1.], [1., 1.]])
    assert model.score(x, np.array([1., 2., 3.]), weights=np.ones(3)) == 0.


def test_loading_reversed_keys_does_not_mutate_input():
    basis = bspline.BSplineBasis(composition.ChemicalSystem(['Ne', 'Xe']))
    model = least_squares.WeightedLinearModel(basis)
    model.coefficients = np.arange(basis.n_feats, dtype=float)
    dump = model.as_dict()
    dump['coefficients'][('Xe', 'Ne')] = dump['coefficients'].pop(('Ne', 'Xe'))
    expected = model.coefficients.copy()
    model.load(solution=dump)
    np.testing.assert_array_equal(model.coefficients, expected)
    assert ('Ne', 'Xe') not in dump['coefficients']


def test_json_roundtrip_strings_and_empty_arrays(tmp_path):
    value = {'note': 'a"b\\c\n\t', 'empty': [], ('W', 'W'): np.arange(3.)}
    text = json_io.dump_interaction_map(value)
    assert json.loads(text)['note'] == value['note']
    path = tmp_path / 'model.json'
    path.write_text(text)
    loaded = json_io.load_interaction_map(path)
    assert loaded['empty'].size == 0
    np.testing.assert_array_equal(loaded[('W', 'W')], np.arange(3.))


def test_database_cache_roundtrip_and_slice(tmp_path):
    atoms = [ase.Atoms('W') for _ in range(3)]
    df = io.prepare_dataframe_from_lists(atoms, energies=[1., 2., 3.],
        forces=[np.zeros((1, 3)) for _ in atoms], prefix='a')
    path = tmp_path / 'cache.db'
    io.cache_data(df, str(path), serial=True)
    loaded = io.read_database(str(path), index=slice(None, None, 2))
    assert [a.get_potential_energy() for a in loaded] == [1., 3.]
    assert io.read_database(str(path), index=slice(0, 0)) == []


def test_file_helpers(tmp_path):
    path = tmp_path / 'test.xyz'
    assert io.resolve_name_conflict(str(path)) == 0
    path.write_text('test')
    assert io.identify_paths(str(tmp_path), filename='test.xyz') == [str(path)]


def test_empty_parallel_work():
    assert parallel.split_dataframe(pd.DataFrame(), 2) == []
    assert parallel.split_zip(2, []) == [[]]
    with ThreadPoolExecutor() as client:
        assert parallel.gather_and_merge([], client=client, progress=False) == []


def test_all_elements_are_supported():
    from ase.data import chemical_symbols
    assert len(composition.ChemicalSystem(chemical_symbols[1:]).element_list) == 118


def test_directional_displacements_keep_each_force_sign():
    atoms = ase.Atoms('W2', positions=[[0., 0., 0.], [3., 0., 0.]])
    forces = np.array([[1., -1., 1.], [-1., 1., -1.]])
    snapshots, _ = geometry.generate_displacements_from_forces(atoms, 0., forces,
                                                               d=.01, random=False)
    for direction in range(3):
        for i in range(2):
            diff = snapshots[direction * 2 + i].positions - atoms.positions
            assert diff[i, direction] == pytest.approx(.01 * np.sign(forces[i, direction]))
