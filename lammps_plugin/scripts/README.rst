This document describes how to generate UF3 lammps potential from an UF3 model file(JSON).

Usuage-

.. code:: bash

   python generate_uf3_lammps_pots.py -m <model_name> -d <directory> -a <AUTHOR> 

Will create :code:`directory` if it does not exists and will write UF3 lammps potential files in it. 


It will also additionally print lines that should be added to the lammps input file for using UF3 lammps potential files.
 
