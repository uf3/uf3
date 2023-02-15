This document describes how to use UF3 potentials in `lammps <https://www.lammps.org/>`_ MD code. See the `tungsten_example_ <https://github.com/monk-04/uf3/tree/lammps_implementation/lammps_plugin/tungsten_example>`_ directory for an example of lammps input file and UF3 lammps potential files.

Compiling lammps with UF3 library
-----

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


Running lammps with UF3 potential
-----

To use UF3 potentials in lammps just add the following tags to the lammps input file-

.. code:: bash

    pair_style uf3 3 1
    pair_coeff * * W_W W_W_W

The 'uf3' keyword in :code:`pair_style` invokes the UF3 potentials in lammps. The number next to the :code:`uf3` keyword tells lammps whether the user wants to run the MD code with just 2-body or 2 and 3-body UF3 potentials. The last number of this line specifies the number of elemnts in the system. So in the above example, the user wants to run MD simulation with UF3 potentials containing both 2-body and 3-body interactions on a system containing only 1 element.

The :code:`pair_coeff` tag is used to read in the user-provided UF3 lammps potential files. These files can be generated directly from the :code:`json` potential files of UF3. The two asterisks on this line are not used in the current implementation but should be present. After the asterisks list all the 2 and 3-body UF3 lammps potential files seperated by space. Make sure these files are present in the current run directory or in directories where lammps can find them.

Structure of UF3 lammps potential file
-----

This section describes the format of the UF3 lammps potential file. Not following the format can lead to unexpected error in the MD simulation and sometimes unexplained core dumps.

We recommend using the :code:`generate_uf3_lammps_pots.py` script (`found here <https://github.com/monk-04/uf3/tree/lammps_implementation/lammps_plugin/scripts>`_) for generating the UF3 lammps potential files.

2-body potential
====

The 2-body UF3 lammps potential file should have the following format-

.. code:: bash

    #UF3 POT
    2B
    1 1 Rij_CUTOFF NUM_OF_KNOTS
    BSPLINE_KNOTS
    NUM_OF_COEFF
    COEFF
    #

The first line of all UF3 lammps potential files should start with :code:`#UF3 POT` characters. The next line indicates whether the file contains UF3 lammps potential data for 2-body (:code:`2B`) or 3-body (:code:`3B`) interaction.

For 2-body UF3 lammps potential, the first two characters of the third line indicates the atom-types for which this file containes the interaction potential. In the above example the file describes the interaction between atom-type 1 and 1.
The :code:`Rij_CUTOFF` sets the 2-body cutoff for the interaction described by the potential file. :code:`NUM_OF_KNOTS` is the number of knots (or the length of the knot vector) present on the very next line. The :code:`BSPLINE_KNOTS` line should contain all the knots in increasing order. :code:`NUM_OF_COEFF` is the number of coefficients in the :code:`COEFF` line. All the numbers in the BSPLINE_KNOTS and COEFF line should be space-separated. 

3-body potential
====

The 3-body UF3 lammps potential file has a format similar to the 2-body potential file-

.. code:: bash

    #UF3 POT
    3B
    1 2 3 Rjk_CUTOFF Rik_CUTOFF Rij_CUTOFF NUM_OF_KNOTS_JK NUM_OF_KNOTS_IK NUM_OF_KNOTS_IJ
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


The first line is similar to the 2-body potential file and the second line has :code:`3B` characters indicating that this file describes 3-body interaction. The first 3 characters of the third line state the atom type for which this 3-body potential should be used. In the above example, the file will be used to describe the interaction between atom-types 1(i), 2(j), and 3(k). The cutoff distance between atom-type 1 and 2 is :code:`Rij_CUTOFF`, atom-type 1 and 3 is :code:`Rik_CUTOFF` and between 2 and 3 is :code:`Rjk_CUTOFF`. **Note the current implementation works only for UF3 potentials with cutoff distances for 3-body interactions that follows** :code:`2Rij_CUTOFF=2Rik_CUTOFF=Rjk_CUTOFF` **relation.**

The :code:`BSPLINE_KNOTS_FOR_JK`, :code:`BSPLINE_KNOTS_FOR_IK`, and :code:`BSPLINE_KNOTS_FOR_IJ` lines (note the order) contain the knots in increasing order for atoms J and K, I and K, and atoms I and J respectively. The number of knots is defined by the :code:`NUM_OF_KNOTS_*` characters in the previous line.
The shape of the coefficient matrix is defined on the :code:`SHAPE_OF_COEFF_MATRIX[I][J][K]` line followed by the columns of the coefficient matrix, one per line, as shown above. For example, if the coefficient matrix has the shape of 8x8x13, then :code:`SHAPE_OF_COEFF_MATRIX[I][J][K]` will be :code:`8 8 13` followed by 64 (8x8) lines each containing 13 coefficients seperated by space.

All the UF3 lammps potential files end with :code:`#` character.