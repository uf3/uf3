# Ultra-Fast Force Fields (UF<sup>3</sup>)

[![Tests](https://github.com/sxie22/fast-linear-qmml/workflows/Tests/badge.svg)](https://github.com/sxie22/fast-linear-qmml/actions)

## Setup
```bash
conda create --name uf3_env python=3.7
conda activate uf3_env
conda install -c anaconda -c conda-forge mpi4py lammps
pip install pylammpsmpi
git clone https://github.com/sxie22/uf3
cd uf3
pip install wheel
pip install -r requirements.txt
pip install -e .
```

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

## Basic Usage

## Citing This Work

