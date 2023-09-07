This document describes how to generate UF3 lammps potential from an input structure file(`POSCAR <https://www.vasp.at/wiki/index.php/POSCAR>`_), and UF3 model file(JSON).

Usuage-

.. code:: bash

   python generate_uf3_lammps_pots.py NAME_of_UF3_model_File DIR_PATH_or_NAME 

Will create :code:`DIR_PATH_or_NAME` if it does not exists and will write UF3 lammps potential files to :code:`DIR_PATH_or_NAME`. Make sure NAME_of_POSCAR and NAME_of_UF3_model_File are present in the same directory from which :code:`generate_uf3_lammps_pots.py` is executed.

A lammps structure file with the name :code:`lammps.struct` will also be generated from the provided POSCAR file. This structure file should be used for running the lammps simulation. The potential files produced can also be used with other structures. But care must be taken to ensure that the elemental species map onto the right species number in the lammps structure file. 


It will also additionally print lines that should be added to the lammps input file for using UF3 lammps potential files.
 
