# Repository audit fixes

- Repository: `jatinrawatjr/uf3_Jatin` (fork of `uf3/uf3`)
- Branch: `fix/upstream-audit`
- Upstream base commit: `bc3279d9f986bd0b1f3720764af5196d989bea40` (`develop`)
- Original audit commit: `cafe2f74d1abbefe84966d9463def93486febf07`
- Date: 20 October 2026

This file was created before implementation and updated as the fixes were
completed. The numbers below correspond to the original repository audit.
The original audit base and the upstream base have identical tracked contents;
the fixes transferred without conflicts. Only this transfer metadata changed.

## What changed, where, why, and how

| Audit | Where | Why the change was needed | How it was fixed |
| --- | --- | --- | --- |
| 1 | `uf3/data/io.py` | Copying ASE atoms discarded the attached calculator before energy and force labels were read. | Temporarily retain the calculator on copied atoms while extracting results, pass the actual geometry to the calculator, then detach it from the copy. The original atoms are not modified. |
| 2 | `uf3/representation/bspline.py` | Mutable default regularization dictionaries let one fit change later fits. | Create new dictionaries for every call and copy any supplied mappings before adding overrides. |
| 3 | `uf3/data/geometry.py` | A zero nonperiodic vector made valid slabs and chains lose periodic neighbors. | Calculate replication factors from the reciprocal basis of the periodic vectors only. Reject zero or dependent periodic vectors. |
| 4 | `uf3/regression/least_squares.py`, `uf3/representation/process.py` | Missing or constant energy/force blocks could divide by zero and produce nonfinite coefficients or legacy training weights. | Ignore empty variance batches, give missing blocks zero contribution, use finite scales for constant targets, explicitly select active fitting terms, and reject empty training selections or nonfinite fitting matrices/solutions. |
| 5 | `uf3/util/user_config.py` | The YAML loader accepted Python object tags. | Use `yaml.safe_load`, require mapping-shaped configuration sections, and merge defaults without executing arbitrary Python constructors. |
| 6 | `uf3/data/io.py` | pandas 3 removed `delim_whitespace` from `read_csv`. | Use `sep=r'\s+'` in the LAMMPS and TSV readers. |
| 7 | `uf3/data/io.py` | Modern pandas removed `Series.iteritems`. | Iterate using `Series.items`; verify exported energies survive a database round trip. |
| 8 | `setup.py`, `MANIFEST.in` | Built installations omitted `uf3/default_options.yaml`. | Declare the YAML file as package data and include it in the source archive. Resolve requirements relative to `setup.py`, and remove the obsolete `tests_require` setup option. |
| 9 | `uf3/util/user_config.py`, `uf3/representation/bspline.py` | Configuration omitted defaults, lost basis context, passed duplicate feature arguments, and treated a model filename as coefficient data. | Deep-merge supported defaults, supply element/degree context to the basis, accept dash-separated YAML interaction keys, honor optional trim settings, read nested feature options, call `load(filename=...)`, avoid mutating caller settings, and warn if a handler cannot be initialized. |
| 10 | `uf3/forcefield/calculator.py` | ASE stress requested a force-consistent energy property that was not advertised. | Expose `free_energy` and use the finite-difference stress function, with a fallback for older ASE releases. |
| 11 | `uf3/forcefield/calculator.py`, `requirements.txt` | Relaxation used an obsolete filter location and optimizer argument and performed an extra optimizer step. | Use the available ASE cell filter, remove the obsolete argument and duplicate step, and allow an explicit step limit. Set the supported ASE minimum to 3.23. |
| 12 | `uf3/forcefield/properties/phonon.py`, `requirements.txt` | Phonopy removed legacy getters/setters; the old implementation also mixed Cartesian offsets into scaled positions and derived paths from a supercell. | Use supported attributes/result objects, preserve scaled coordinates, derive the band path from the primitive cell with atomic numbers, and declare `phonopy>=2.26,<5`. Update plotting colormap lookup. |
| 13 | `lammps_plugin/scripts/generate_uf3_lammps_pots.py` | Unused pymatgen imports blocked startup, trim dictionaries were written instead of integers, and the printed command had an extra argument. | Remove the unused imports, write each interaction's scalar trim settings, and print `pair_style uf3 BODY_ORDER`. |
| 14 | `uf3/util/subsample.py` | Tied distances selected the same row repeatedly, multidimensional input could select an invalid flattened index, and empty input crashed. | Start from a valid row, exclude selected row indices explicitly, handle empty arrays, and avoid building a distance matrix when returning all rows without sampling. |
| 15 | `uf3/regression/least_squares.py` | Weighted scoring used feature count where sample count was required. | Apply the existing validated sample-weight broadcasting helper instead of a feature-sized matrix. |
| 16 | `uf3/regression/least_squares.py` | Normalizing reversed interaction keys modified the dictionary during iteration. | Normalize into a new dictionary. Also fill missing three-body interaction vectors consistently with the existing pair-vector behavior. |
| 17 | `uf3/util/json_io.py` | Strings containing quotes/backslashes produced invalid JSON; empty lists were indexed at position zero. | Use standard JSON string escaping and guard empty lists in both encoding and decoding. |
| 18 | `uf3/data/io.py` | Custom geometry and energy column names were replaced by hard-coded defaults during synchronization. | Forward `atoms_key` and `energy_key` consistently; use `set_array` so explicit forces replace old component arrays. |
| 19 | `.github/workflows/tests.yml` | CI tried to install system packages without privileges and did not check installed resources, optional integration, or documentation. | Remove unnecessary system installation and obsolete runner workaround; run Python 3.9–3.12 tests, distribution builds, installed-resource checks, and a separate optional-phonon/documentation job. |
| 20 | `.readthedocs.yaml`, `docs/requirements.txt`, `docs/conf.py`, `docs/index.rst`, `scripts/options.rst`, affected API docstrings | Documentation selected an unsupported Python version and emitted malformed reStructuredText diagnostics. | Select Python 3.12 using current Read the Docs keys, simplify documentation-only dependencies, repair the API navigation and malformed docstrings/lists, remove references to absent static assets, and enforce a warning-free HTML build. |

The documentation corrections affect docstrings in `uf3/data/io.py`,
`uf3/forcefield/properties/elastic.py`, `uf3/regression/{least_squares,optimize,regularize}.py`,
`uf3/representation/{angles,bspline,distances,process}.py`, and `uf3/util/plotting.py`.

## Smaller audit cases also fixed

- `uf3/data/io.py`: return zero for a missing-file name-conflict query; join a
  discovered relative filename to its directory; honor positive and negative
  database slice steps and return an empty list for an empty slice.
- `uf3/util/parallel.py`: return empty results for empty split/gather inputs.
- `uf3/data/composition.py`: restore the six omitted element symbols Po–Ac.
- `uf3/data/geometry.py`: retain the requested displacement magnitude when
  computing each Cartesian direction instead of modifying it between axes.
- `.gitignore`: exclude generated package builds, test cache, and generated
  documentation so they are not accidentally committed.

## Regression coverage

`tests/test_audit_regressions.py` tests the data-loss cases, custom columns,
independent regularization calls, chain/slab invariance, invalid periodic cells,
constant/missing target blocks, HDF5 fitting, configuration construction,
unsafe YAML rejection, stress, both relaxation modes, optional phonopy,
the exporter CLI/file headers, sampling, scoring, model/JSON round trips,
database caching/slices, file helpers, empty parallel work, element coverage,
and directional displacement signs.

The first regression run, before implementation, had **35 failures and 1 pass**.
An additional legacy-weight regression was added after inspecting the original
suite's divide-by-zero warnings.

## Validation

Completed local checks:

- Full suite with pandas 2.2.3: **150 passed** before the extra legacy-weight test.
- Final full suite with pandas 3.0.6, ASE 3.29.0, phonopy 4.8.3:
  **151 passed**, 4 existing warning emissions (custom-knot notices and
  deprecated legacy training API).
- Final full suite with the supported minimum ASE 3.23.0 and phonopy 2.26.0,
  loaded from a separate dependency directory: **151 passed**, 22 warning
  emissions, including deprecations in those older dependencies.
- Numerical energy/force consistency: **12 configurations passed**, covering
  unary/binary chemistry, degree 2/3, and nonperiodic/full/partial periodic cells;
  maximum force finite-difference discrepancy below **5 × 10⁻⁹**.
- Wheel and source archive both built and contain `uf3/default_options.yaml`.
- Outside the source checkout, the final installed wheel loads default
  settings, honors a custom `W-W` cutoff, and initializes basis, feature,
  and learning handlers successfully.
- Documentation HTML build with `-W --keep-going`: **passed without warnings**.
- Targeted fatal-error lint and whitespace checks (respecting existing CRLF
  files): passed.
- Dependency consistency: `pip check` passed.

The reproducible check commands are:

```bash
python -m pytest -q
python -m build
ruff check uf3 tests setup.py lammps_plugin/scripts --select E9,F63,F7,F82
sphinx-build -E -W --keep-going -b html docs /tmp/uf3-docs
```

The installed-wheel check is run outside the repository after installation,
so imports cannot silently fall back to the source tree. Hosted CI results
are reported separately after pushing the branch.

## Limits

Local checks use Python 3.12. GitHub Actions is configured to test Python
3.9–3.12; its remote results must be assessed separately. Full scientific
notebook workloads, elastic-constant calculations, and native UF3 LAMMPS,
MPI, and Kokkos simulations were not run. The local LAMMPS executable predates
the UF3 pair style; exporter startup and file content were tested instead.

## Commit attribution

Author and committer: `jatinrawatjr`, using the existing Git email identity.
The commit message has no Codex/OpenAI co-author trailer.
