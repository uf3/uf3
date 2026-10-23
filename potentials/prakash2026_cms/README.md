# UF<sup>3</sup> potentials for Ni, Cu, Li, Mo, Si, and Ge

These are the six single-element UF<sup>3</sup> potentials (two- and three-body B-splines) from

> P. Prakash, S. Dong, and R. G. Hennig, "Benchmarking of fast and interpretable UF machine learning potentials", *Computational Materials Science* (2026). https://doi.org/10.1016/j.commatsci.2026.115112 (preprint: https://arxiv.org/abs/2608.27277)

Please cite this paper if you use them.

The potentials were fit to the DFT data of Zuo et al., *J. Phys. Chem. A* 124, 731 (2020), using the 90:10 train/test splits distributed with [mlearn](https://github.com/materialyzeai/mlearn/tree/master/data). The training data are not copied here. The hyperparameters for each element are listed in Table 1 of the paper.

## Files

Each element folder `X/` (X = Ni, Cu, Li, Mo, Si, Ge) contains

| File | Description |
|---|---|
| `X.uf3` | LAMMPS potential in the single-file ML-UF3 format (LAMMPS 29Aug2024 or later) |
| `X_model.json` | The uf3 model (knots, spline coefficients, and one-body offset) |
| `X_X`, `X_X_X` | The same potential in the older two-file format (2-body and 3-body), used with `pair_style uf3 3 1` for the simulations in the paper |

`X.uf3` was made from `X_X` and `X_X_X` by rewriting only the header lines. All knots and coefficients are copied unchanged. Forces from LAMMPS with `X.uf3` agree with the uf3 Python calculator to within 5 × 10⁻⁶ eV/Å on rattled 2×2×2 supercells for all six elements.

## Usage

LAMMPS (built with the ML-UF3 package), metal units:

```
pair_style uf3 3
pair_coeff * * Ni.uf3 Ni
```

Python, through the ASE calculator in uf3:

```python
from uf3.regression.least_squares import WeightedLinearModel
from uf3.forcefield.calculator import UFCalculator

model = WeightedLinearModel.from_json("Ni/Ni_model.json")
atoms.calc = UFCalculator(model)  # atoms is an ase.Atoms object
```

## Which potentials to use

Melting points are from two-phase coexistence simulations (Table 2 of the paper). The other notes are from Section 3.4 of the paper.

| Element | UF<sup>3</sup> melting point (K) | Experiment (K) | Notes |
|---|---|---|---|
| Ni | 1832 ± 10 | 1728 | Melting point 6% high. Thermal expansion 25–30% low. |
| Cu | 1427 ± 8 | 1358 | Melting point 5% high. Thermal expansion 15–20% low. |
| Li | 455 ± 12 | 454 | Melting point within 1% of experiment. |
| Mo | 1689 ± 13 | 2896 | Melting point 42% low. Use only below 1689 K. |
| Si | 1352 ± 6 | 1687 | Melting point 20% low. Phonons up to 17% too soft. |
| Ge | – | 1211 | **Not stable in MD. Do not use for simulations.** It is included only so that the test errors in the paper can be reproduced. |
