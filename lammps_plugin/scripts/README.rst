This document describes how to generate UF3 lammps potential from an UF3 model file(JSON).

Usuage-

.. code:: bash

   python generate_uf3_lammps_pots.py NAME_of_UF3_model_File DIR_PATH_or_NAME 

Will create :code:`DIR_PATH_or_NAME` if it does not exists and will write UF3 lammps potential files to :code:`DIR_PATH_or_NAME`. 


It will also additionally print lines that should be added to the lammps input file for using UF3 lammps potential files.
 
