This document describes how to use UF3 potentials in `LAMMPS <https://www.lammps.org/>`_ MD code. The documentation for :code:`pair_style uf3` is available `here <https://docs.lammps.org/pair_uf3.html>`_.

UF3 LAMMPS has been integrated with the official LAMMPS package available at- https://github.com/lammps/lammps/. It is the stable version of UF3 LAMMPS.

The development version of UF3 LAMMPS is available at- https://github.com/uf3/lammps

The stable version and development version of UF3 LAMMPS have the following restrictions-

1. The current UF3 LAMMPS version restricts the leading and trailing trims to 0 and 3, respectively. If you developed a UF3 potential with different trims the LAMMPS code will error out. If you need support for different trims, please open an issue.


2. The current UF3 LAMMPS version restricts the 3-body cutoffs to follow- :code:`2rij=2rik=rjk`. For example, in the Tungsten :code:`uf23_potential_demo.ipynb` `example <https://github.com/uf3/uf3/blob/develop/examples/tungsten_extxyz/uf23_potential_demo.ipynb>`_ the :code:`r_max_map` value for :code:`("W", "W", "W")` can be :code:`[4,4,8]`, :code:`[2,2,4]`, :code:`[3,3,6]`, but it cannot be :code:`[4,4,7]`. You can fit a UF3 potential with code:`r_max_map` equal to :code:`[4,4,7]` but you won't be able to use the potential in UF3 LAMMPS.

.. contents:: Contents
   :depth: 1
   :local: 

=================================
Compiling lammps with UF3 library
=================================

The best source of information about compiling LAMMPS with different packages is the official documentation available at this `link <https://docs.lammps.org/Build.html>`_.

Here we provide only minimal instructions for convenience.

Get the latest version of UF3 LAMMPS-
.. code:: bash

   git clone https://github.com/lammps/lammps.git
   cd lammps


If compiling with :code:`CMake`, make a :code:`build` directory and compile LAMMPS with :code:`PKG_ML-UF3`

.. code:: bash

   cd lammps
   mkdir build
   cmake ../cmake/ -D PKG_ML-UF3=yes
   cmake --build .

This will compile lammps with support for UF3 potentials. For more information on how to build lammps see this `link <https://docs.lammps.org/Build.html>`_.


If you want to compile with MPI support enabled, you additionally have to set the BUILD_MPI flag and make sure that a suitable MPI runtime can be found by the compiler. To make the executables distinguishable, the `LAMMPS_MACHINE` option sets a suffix for the output file.

.. code:: bash

   cd lammps/build
   cmake ../cmake/ -D PKG_ML-UF3=yes -D BUILD_MPI=yes -D LAMMPS_MACHINE=mpi # Compiles to lmp_mpi
   cmake --build .

Kokkos
======
To compile lammps with support for the kokkos accelerator variants of uf3:

.. code:: bash

   mkdir lammps/build_kokkos
   cd build_kopkkos
   cmake -D LAMMPS_EXCEPTIONS=ON -D BUILD_SHARED_LIBS=ON -D PKG_KOKKOS=yes KOKKOS_FLAGS -D PKG_ML-UF3=ON -D CMAKE_POSITION_INDEPENDENT_CODE=ON -D CMAKE_EXE_FLAGS="-dynamic" ../cmake

where :code:`KOKKOS_FLAGS` are machine specific flags. For example when compiling for Nvidia A100 GPUs-

.. code:: bash

   cmake -D LAMMPS_EXCEPTIONS=ON -D BUILD_SHARED_LIBS=ON -D PKG_KOKKOS=yes -D Kokkos_ARCH_AMPERE80=ON -D Kokkos_ENABLE_CUDA=yes -D PKG_ML-UF3=ON -D CMAKE_POSITION_INDEPENDENT_CODE=ON -D CMAKE_EXE_FLAGS="-dynamic" ../cmake

Refer to `lammps documentation <https://docs.lammps.org/Speed_kokkos.html>`_ for more details.

=================================
Running lammps with UF3 potential
=================================

Refer to the :code:`pair_style uf3` LAMMPS page for details on this topic at `here <https://docs.lammps.org/pair_uf3.html>`_.

**The old style of listing all UF3 LAMMPS potential files after pair_coeff * * in single line is deprecated**

See the `tungsten_example <https://github.com/uf3/uf3/tree/develop/lammps_plugin/tungsten_example>`_ directory for an example of lammps input file and UF3 lammps potential file.

Use :code:`generate_uf3_lammps_pots.py` to generate the UF3 LAMMPS potential file from the UF3 :code:`json` potential file located in the :code:`scripts` directory.
