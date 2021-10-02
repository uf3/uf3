Ultra-Fast Force Fields (UF3)
=============================

|Tests|

All-atom dynamics simulations have become an indispensable quantitative
tool in physics, chemistry, and materials science, but large systems and
long simulation times remain challenging due to the trade-off between
computational efficiency and predictive accuracy. The UF3 framework is
built to address this challenge by combinining effective two- and
three-body potentials in a cubic B-spline basis with regularized linear
regression to obtain machine-learning potentials that are physically
interpretable, sufficiently accurate for applications, and as fast as
the fastest traditional empirical potentials.

Documentation: https://uf3.readthedocs.io/

This repository is still under construction. Please feel free to open
new issues for feature requests and bug reports.

Setup
-----

.. code:: bash

   conda create --name uf3_env python=3.7
   conda activate uf3_env
   git clone https://github.com/uf3/uf3.git
   cd uf3
   pip install wheel
   pip install -r requirements.txt
   pip install numba
   pip install -e .

Getting Started
---------------

Please see the examples in uf3/examples/tungsten_extxyz for basic usage.

Overviews for individual modules can be found in uf3/examples/modules
(WIP).

Standalone scripts and configuration generators/parsers are in
development.

Optional Dependencies
---------------------

Elastic constants:

::

   pip install setuptools_scm
   pip install "elastic>=5.1.0.17"

Phonon spectra:

::

   pip install spglib
   pip install seekpath
   pip install "phonopy>=2.6.0"

LAMMPS interface:

::

   conda install numpy==1.20.3 --force-reinstall
   conda install -c conda-forge lammps --no-update-deps

Dependencies
------------

-  We rely on ase to handle parsing outputs from atomistic codes like
   LAMMPS, VASP, and C2PK.
-  We use Pandas to keep track of atomic configurations and their
   energies/forces as well as organizing data for featurization and
   training.
-  B-spline evaluations use scipy, numba, and ndsplines.
-  PyTables is used for reading/writing HDF5 files.

Citing This Work
----------------

The manuscript is still in preparation.

.. |Tests| image:: https://github.com/uf3/uf3/workflows/Tests/badge.svg
   :target: https://github.com/uf3/uf3/actions
