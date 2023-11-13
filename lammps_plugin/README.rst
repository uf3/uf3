This document describes how to use UF3 potentials in `lammps <https://www.lammps.org/>`_ MD code. See the `tungsten_example <https://github.com/monk-04/uf3/tree/lammps_implementation/lammps_plugin/tungsten_example>`_ directory for an example of lammps input file and UF3 lammps potential files.

.. contents:: Contents
	:depth: 1
	:local: 

=====
Compiling lammps with UF3 library
=====

Before running lammps with UF3 potentials, lammps must be re-compiled with the :code:`pair_uf3` and other supporting libraries contained in :code:`ML-UF3` directory.

Copy the :code:`ML-UF3` directory to the appropriate lammps source code directory.

.. code:: bash

   mv ML-UF3 LAMMPS_BASE_DIR/src/.

If compiling lammps with :code:`CMake`, add the :code:`ML-UF3` keyword to :code:`set(STANDARD_PACKAGES)` list in :code:`LAMMPS_BASE_DIR/cmake/CMakeLists.txt` file (search for :code:`ML-SNAP` and add :code:`ML-UF3` in the very next line). Go to the :code:`build` directory in :code:`LAMMPS_BASE_DIR` (make one if it doesn't exist) and compile lammps.

.. code:: bash

 cd LAMMPS_BASE_DIR/build
 cmake ../cmake/ -D PKG_ML-UF3=yes
 cmake --build .

This will compile lammps with support for UF3 potentials. For more information on how to build lammps see this link_.

.. _link: https://docs.lammps.org/Build.html

If you want to compile with MPI support enabled, you additionally have to set the BUILD_MPI flag and make sure that a suitable MPI runtime can be found by the compiler. To make the executables distinguishable, the `LAMMPS_MACHINE` option sets a suffix for the output file.

.. code:: bash

 cd LAMMPS_BASE_DIR/build
 cmake ../cmake/ -D PKG_ML-UF3=yes -D BUILD_MPI=yes -D LAMMPS_MACHINE=mpi # Compiles to lmp_mpi
 cmake --build .

Kokkos
=====
The kokkos accelerator variants of uf3 (i.e. :code:`pair_style uf3/kk` with GPU support) is under active development. For an early access refer to the :code:`README` found here_.

.. _here: https://github.com/monk-04/uf3/tree/lammps_implementation_v2/lammps_plugin#kokkos

=====
Running lammps with UF3 potential
=====

**The old style of listing all UF3 LAMMPS potential files after pair_coeff * * in single line is deprecated**

To use UF3 potentials in lammps just add the following tags to the lammps input file-

.. code:: bash

    pair_style uf3 3 1
    pair_coeff 1 1 W_W 
    pair_coeff 1 1 1 W_W_W

The 'uf3' keyword in :code:`pair_style` invokes the UF3 potentials in lammps. The number next to the :code:`uf3` keyword tells lammps whether the user wants to run the MD code with just 2-body or 2 and 3-body UF3 potentials. The last number of this line specifies the number of elemnts in the system. So in the above example, the user wants to run MD simulation with UF3 potentials containing both 2-body and 3-body interactions on a system containing only 1 element.

The :code:`pair_coeff` tag is used to read in the user-provided UF3 lammps potential files. These files can be generated directly from the :code:`json` potential files of UF3. We recommend using the :code:`generate_uf3_lammps_pots.py` script (`found here <https://github.com/monk-04/uf3/tree/lammps_implementation/lammps_plugin/scripts>`_) for generating the UF3 lammps potential files. It will also additionally print lines that should be added to the lammps input file for using UF3 lammps potential files.

After :code:`pair_coeff` specify the interactions (two numbers for 2-body, three numbers for 3-body) followed by the name of the potential file. The user can also use asterisks:code:`*` for wild-card characters. In this case the behaviour is similar to other LAMMPS :code:`pair_style` for example LJ. The user can also specify. Make sure these files are present in the current run directory or in directories where lammps can find them.

As an example for a multicomponet system containing elements 'A' and 'B' the above lines can be-

.. code:: bash

   pair_style uf3 3 2
   pair_coeff 1 1 A_A
   pair_coeff 1 2 A_B
   pair_coeff 2 2 B_B
   pair_coeff 1 1 1 A_A_A
   pair_coeff 1 1 2 A_A_B
   pair_coeff 1 2 2 A_B_B
   pair_coeff 2 1 1 B_A_A
   pair_coeff 2 1 2 B_A_B
   pair_coeff 2 2 2 B_B_B

Following format is also a valid for system containing elements 'A' and 'B'

.. code:: bash

   pair_style uf3 3 2
   pair_coeff * * A_A
   pair_coeff 1 * * A_A_A
   pair_coeff 2 * * B_B_B

   
Alternatively, if the user wishes to use only the 2-body interactions from a model containing both two and three body interaction simply change the number next to :code:`uf3` to :code:`2` and don't list the three body interaction files in the :code:`pair_coeff` line. Beware! Using only the 2-body interaction from a model containing both 2 and 3-body is not recommended and will give wrong results!

.. code:: bash
  pair_style uf3 2 2
  pair_coeff 1 1 A_A
  pair_coeff 1 2 A_B
  pair_coeff 2 2 B_B
  
=====
Structure of UF3 lammps potential file
=====

This section describes the format of the UF3 lammps potential file. Not following the format can lead to unexpected error in the MD simulation and sometimes unexplained core dumps.


2-body potential
====

**The old UF3 LAMMPS potential files can still be used but a warning is printed**

The 2-body UF3 lammps potential file should have the following format-

.. code:: bash

    #UF3 POT
    2B LEADING_TRIM TRAILING_TRIM TYPE_OF_KNOT_SPACING
    Rij_CUTOFF NUM_OF_KNOTS
    BSPLINE_KNOTS
    NUM_OF_COEFF
    COEFF
    #

The first line of all UF3 lammps potential files should start with :code:`#UF3 POT` characters. The next line indicates whether the file contains UF3 lammps potential data for 2-body (:code:`2B`) or 3-body (:code:`3B`) interaction. This is followed by :code:`LEADING_TRIM` and :code:`TRAILING_TRIM` number. The current implementation is only tested for :code:`LEADING_TRIM=0` and :code:`TRAILING_TRIM=3`. If other values are used LAMMPS will stop with an error message. The :code:`TYPE_OF_KNOT_SPACING` specifies if the spacing between the knots is constant :code:`uk` (uniform-knots/linear-knots) or is non-uniform :code:`nk`.

The :code:`Rij_CUTOFF` sets the 2-body cutoff for the interaction described by the potential file. :code:`NUM_OF_KNOTS` is the number of knots (or the length of the knot vector) present on the very next line. The :code:`BSPLINE_KNOTS` line should contain all the knots in ascending order. :code:`NUM_OF_COEFF` is the number of coefficients in the :code:`COEFF` line. All the numbers in the BSPLINE_KNOTS and COEFF line should be space-separated. 

3-body potential
====

**The old UF3 LAMMPS potential files can still be used but a warning is printed**

The 3-body UF3 lammps potential file has a format similar to the 2-body potential file-

.. code:: bash

    #UF3 POT
    3B LEADING_TRIM TRAILING_TRIM TYPE_OF_KNOT_SPACING
    Rjk_CUTOFF Rik_CUTOFF Rij_CUTOFF NUM_OF_KNOTS_JK NUM_OF_KNOTS_IK NUM_OF_KNOTS_IJ
    BSPLINE_KNOTS_FOR_JK
    BSPLINE_KNOTS_FOR_IK
    BSPLINE_KNOTS_FOR_IJ
    SHAPE_OF_COEFF_MATRIX[I][J][K]
    COEFF_MATRIX[0][0][K]
    COEFF_MATRIX[0][1][K]
    COEFF_MATRIX[0][2][K]
    .
    .
    .
    COEFF_MATRIX[1][0][K]
    COEFF_MATRIX[1][1][K]
    COEFF_MATRIX[1][2][K]
    .
    .
    .
    #


The first line is similar to the 2-body potential file and the second line has :code:`3B` characters indicating that this file describes 3-body interaction. The :code:`3B` is followed by :code:`LEADING_TRIM` and :code:`TRAILING_TRIM` number. The current implementation is only tested for :code:`LEADING_TRIM=0` and :code:`TRAILING_TRIM=3`. If other values are used LAMMPS will stop with an error message. The :code:`TYPE_OF_KNOT_SPACING` specifies if the spacing between the knots is constant :code:`uk` (uniform-knots/linear-knots) or is non-uniform :code:`nk`.

Similar to the 2-body potential file, the third line sets the cutoffs and length of the knots. The cutoff distance between atom-type 1 and 2 is :code:`Rij_CUTOFF`, atom-type 1 and 3 is :code:`Rik_CUTOFF` and between 2 and 3 is :code:`Rjk_CUTOFF`. **Note the current implementation works only for UF3 potentials with cutoff distances for 3-body interactions that follows** :code:`2Rij_CUTOFF=2Rik_CUTOFF=Rjk_CUTOFF` **relation.**

The :code:`BSPLINE_KNOTS_FOR_JK`, :code:`BSPLINE_KNOTS_FOR_IK`, and :code:`BSPLINE_KNOTS_FOR_IJ` lines (note the order) contain the knots in increasing order for atoms J and K, I and K, and atoms I and J respectively. The number of knots is defined by the :code:`NUM_OF_KNOTS_*` characters in the previous line.
The shape of the coefficient matrix is defined on the :code:`SHAPE_OF_COEFF_MATRIX[I][J][K]` line followed by the columns of the coefficient matrix, one per line, as shown above. For example, if the coefficient matrix has the shape of 8x8x13, then :code:`SHAPE_OF_COEFF_MATRIX[I][J][K]` will be :code:`8 8 13` followed by 64 (8x8) lines each containing 13 coefficients seperated by space.

All the UF3 lammps potential files end with :code:`#` character.
