This document describes how to add ML-UF3 unittest to LAMMPS. Refer to `this link <https://docs.lammps.org/Developer_unittest.html>`_ for more information on unittest in LAMMPS.

To run the ML-UF3 unittest, copy the contents of :code:`unittest` directory (i.e :code:`A_A.uf3_pot`, :code:`A_A_A.uf3_pot` and :code:`manybody-pair-uf3.yaml`) to the unittest directory of LAMMPS and re-build LAMMPS-

.. code:: bash

    cp unittest/* LAMMPS_BASE_DIR/unittest/force-styles/tests/
    cd LAMMPS_BASE_DIR/build
    cmake ../cmake/ -D PKG_ML-UF3=yes -D ENABLE_TESTING=on
    cmake --build .

After the build is sucessful go to the LAMMPS unittest directory for pair style and run-

.. code:: bash

    cd LAMMPS_BASE_DIR/unittest/force-styles/tests/
    LAMMPS_BASE_DIR/build/test_pair_style manybody-pair-uf3.yaml


The current implementation will only pass 2 out of 8 tests, remaining 6 tests will be skipped.
