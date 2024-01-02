This document describes how to featruize data using the Ultra Fast Featurization module.


In UF3, featurization is the slowest step. A solution to this problem is to perform all the heavy computation using a precompiled C++ library.


Dependencies
-----

In order to use this feature, the following packages are required-

1. cmake - For compiling and linking all the required libraries
2. pybind - For exposing python types in C++ (`see here <https://github.com/pybind/pybind11>`_)
3. Cpp HDF5 libraries - For writing computed features to HDF5 file


Setup
-----

UF3 needs to be re-installed to enable this feature-

1. First export environment variables

.. code:: bash

   export HDF5_INCLUDE_DIR=/Path/to/HDF5/include/dir
   export HDF5_LIB_DIR=/Path/to/HDF5/lib/dir 
   export ULTRA_FAST_FEATURIZER=True

2. git clone the latest version of uf3 and select UltraFastFeaturization branch

.. code:: bash

   git clone https://github.com/uf3/uf3.git
   cd uf3
   git checkout UltraFastFeaturization
   pip install .
      

