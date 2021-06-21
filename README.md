# Ultra-Fast Force Fields (UF<sup>3</sup>)

[![Tests](https://github.com/sxie22/fast-linear-qmml/workflows/Tests/badge.svg)](https://github.com/sxie22/fast-linear-qmml/actions)


All-atom dynamics simulations have become an indispensable quantitative tool in physics, chemistry, and materials science, but large systems and long simulation times remain challenging due to the trade-off between computational efficiency and predictive accuracy. The UF<sup>3</sup> framework is built to address this challenge by combinining effective two- and three-body potentials in a cubic B-spline basis with regularized linear regression to obtain machine-learning potentials that are physically interpretable, sufficiently accurate for applications, and as fast as the fastest traditional empirical potentials.

This repository is still under construction. Please feel free to open new issues for feature requests and bug reports.

## Setup
```bash
conda create --name uf3_env python=3.7
conda activate uf3_env
git clone https://github.com/sxie22/uf3
cd uf3
pip install wheel
pip install -r requirements.txt
pip install numba
pip install -e .
```

## Getting Started

Please see the examples in uf3/examples/W for basic usage (WIP).

## Optional Dependencies
Elastic constants:
```
pip install setuptools_scm
pip install "elastic>=5.1.0.17"
```

Phonon spectra:
```
pip install seekpath
pip install "phonopy>=2.6.0"
```

LAMMPS interface:
```
conda install -c anaconda -c conda-forge lammps
```

## Dependencies
We rely on ase to handle parsing outputs from atomistic codes like LAMMPS, VASP, and C2PK. We use Pandas to keep track of atomic configurations and their energies/forces as well as organizing data for featurization and training. B-spline evaluations use scipy, numba, and ndsplines.

## Citing This Work
The manuscript is still in preparation.
