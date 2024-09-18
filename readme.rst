Ultra-Fast Force Fields (UF3)
=============================

|Tests|


\Xie, S.R., Rupp, M. & Hennig, R.G., "Ultra-fast interpretable machine-learning potentials", npj Comput Mater 9, 162 (2023). https://doi.org/10.1038/s41524-023-01092-7

.. _https://doi.org/10.1038/s41524-023-01092-7: https://doi.org/10.1038/s41524-023-01092-7

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

.. Recommended: Install UF3 in a new conda environment:

.. .. code:: bash

..    conda create -n uf3_env python=3.8
..    conda activate uf3_env

UF3 can be obtained by cloning the repository and installing it:

..
   1. Download and install automatically from PyPI (recommended):
   pip install uf3
   Download and install manually from GitHub:

.. code:: bash

   git clone https://github.com/uf3/uf3.git
   cd uf3
   pip install .

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
   pip install uf3[elastic_constants]

Phonon spectra:

::

   pip install uf3[phonon_spectra]

LAMMPS interface:

::

   conda install numpy==1.20.3 --force-reinstall
   conda install -c conda-forge lammps --no-update-deps

The environment variable ``$ASE_LAMMPSRUN_COMMAND`` must also be set to use the LAMMPS interface within python. See the `ASE documentation <https://wiki.fysik.dtu.dk/ase/ase/calculators/lammpsrun.html>`_ for details.

Dependencies
------------

-  We rely on ase to handle parsing outputs from atomistic codes like
   LAMMPS, VASP, and CP2K.
-  We use Pandas to keep track of atomic configurations and their
   energies/forces as well as organizing data for featurization and
   training.
-  B-spline evaluations use scipy, numba, and ndsplines.
-  PyTables is used for reading/writing HDF5 files.
-  Matplotlib is used for plotting.
-  We use sklearn for regression.
-  We use tqdm for progress bars.
-  We use plotly for interactive plots.
-  We use PyYaml for configuration files.
-  We use numpy for array operations.


.. |Tests| image:: https://github.com/uf3/uf3/workflows/Tests/badge.svg
   :target: https://github.com/uf3/uf3/actions
