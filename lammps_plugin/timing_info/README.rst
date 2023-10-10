This document describes the timing performance of lammps implementation of UF3.

The tests were performed using a not yet published UF3 potential for the Nb-Sn system. The simulation cell consists of an Nb3Sn structure downloaded from the `Materials Project <materialsproject.org/>`_. The lammps simulations were carried out on the `Hipegator machine <https://help.rc.ufl.edu/doc/Available_Node_Features>`_ using a single AMD EPYC 7702 64-Core processor.

For benchmarking the implementation, the perfomance is compared to simulations performed with LJ potentials. For this comparison a LJ potnetial was fitted to 2-body part of the UF3 potential.

All the simulations utilized perodic boundary conditions and the `NVE fix <https://docs.lammps.org/fix_nve.html>`_. The atoms were first given initial velocities corresponding to 3000K (Gaussian distribution). All the simulations were terminated after 10000 steps.

In the following tables performance of the current implementation is given.

.. list-table:: Nb3Sn 2x2x2 supercell- 64 atoms, 1fs timestep, 1 core
    :header-rows: 1

    * - Potential
      - Speed (timestep/s)
    * - LJ
      - 6513.272
    * - UF2
      - 1163.371
    * - UF3
      - 226.488

.. list-table:: Nb3Sn 4x4x4 supercell- 512 atoms, 1fs timestep, 1 core
    :header-rows: 1

    * - Potential
      - Speed (timestep/s)
    * - LJ
      - 960.561
    * - UF2
      - 115.658
    * - UF3
      - 28.214


.. list-table:: Performance of UF3 with system size, 1fs timestep, 1 core
    :header-rows: 1

    * - Supercell size
      - Speed (timestep/s)
    * - 2x2x2
      - 226.488
    * - 3x3x3
      - 63.358
    * - 4x4x4
      - 28.214

