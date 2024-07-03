This document describes how to generate UF3 lammps potential from an UF3 model file(JSON).

Usuage-

.. code:: bash

   python generate_uf3_lammps_pots.py [-h] -a AUTHOR -u UNITS -m MODEL [-d DIRECTORY] [-k KNOTS_SPACING_TYPE]

   usage: generate_uf3_lammps_pots.py [-h] -a AUTHOR -u UNITS -m MODEL [-d DIRECTORY] [-k KNOTS_SPACING_TYPE]
   
   Generate UF3 LAMMPS potential file
   
   optional arguments:
        -h, --help            show this help message and exit
        -a AUTHOR, --author AUTHOR
                              Author Name Seperated by '_'
        -u UNITS, --units UNITS
                              LAMMPS Units
        -m MODEL, --model MODEL
                              UF3 Model JSON file
        -d DIRECTORY, --directory DIRECTORY
                              Directory path (default: current directory)
        -k KNOTS_SPACING_TYPE, --knots_spacing_type KNOTS_SPACING_TYPE
                              Knot spacing type, uk (uniform spacing) or nk (non-uniform spacing) (default: nk (non-uniform))

Will create :code:`directory` if it does not exists and will write UF3 lammps potential files in it. 


It will also additionally print lines that should be added to the lammps input file for using UF3 lammps potential files.
Citation meta-data will be left blank. Please enter appropriate citation for the generated UF3 LAMMPS potential manually.
