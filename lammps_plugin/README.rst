This document describes how to use UF3 potentials in lammps_ MD code.

.. _lammps: https://www.lammps.org/

Compiling lammps with UF3 library
-----

Before running lammps with UF3 potentials, lammps must be re-compiled with the :code:`pair_uf3` and other supporting libraries contained in :code:`ML-UF3` directory.

Copy the :code:`ML-UF3` directory to the appropriate lammps source code directory.

.. code:: bash

   mv ML-UF3 LAMMPS_BASE_DIR/src/.

If compiling lammps with :code:`CMake`, add the :code:`ML-UF3` keyword to :code:`set(STANDARD_PACKAGES)` list in :code:`LAMMPS_BASE_DIR/cmake/CMakeLists.txt` file. Go the :code:`build` directory in :code:`LAMMPS_BASE_DIR` (make one if it doesn't exist) and compile lammps.

.. code:: bash

 cd LAMMPS_BASE_DIR/build
 cmake ../cmake/ -D PKG_ML-UF3=yes

This will compile lammps with support for UF3 potentials. For more information on how to build lammps see this link_.

.. _link: https://docs.lammps.org/Build.html


Running lammps with UF3 potential
-----

To use UF3 potentials in lammps just add the following tags to the lammps input file-

.. code:: bash

    pair_style uf3 3 1
    pair_coeff * * W_W W_W_W

The 'uf3' keyword in :code:`pair_style` invokes the UF3 potentials in lammps. The number next to the :code:`uf3` keyword on tells lammps whether the user wants to run the MD code with just 2-body or 2 and 3-body UF3 potentials. The last number of this line specifies the number of atoms in the system. So in the above example, the user wants to run MD simulation with UF3 potentials containing both 2-body and 3-body interactions on a system containing only 1 element.

The :code:`pair_coeff` tag is used to read in the user-provided UF3 lammps potential files. These files can be generated directly from the :code:`json` potential files of UF3. In the current implementation, the two asterisks on this line are not used but should be present. After the asterisks comes all the 2 and 3-body UF3 lammps potential files. Make sure these files are present in the current run directory or in directories where lammps can find them.



