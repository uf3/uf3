#!/usr/bin/env python

import os
import setuptools

import re
import subprocess
import sys
from pathlib import Path
from setuptools.command.build_ext import build_ext

module_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(module_dir, 'readme.rst'), "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requires = f.readlines()
    install_requires = requires[:requires.index("EXTRAS_REQUIRE\n")]
    extras_require = {}
    extra = ""
    for extra_require in requires[requires.index("EXTRAS_REQUIRE\n") + 1:]:
        if extra_require.startswith("-"):
            extra = extra_require.strip().lstrip("-")
            continue
        if extra in extras_require:
            extras_require[extra].append(extra_require)
        else:
            extras_require[extra] = [extra_require]

test_requires = ['pytest']

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        if os.environ.get('ULTRA_FAST_FEATURIZER'):

            try:
                out = subprocess.check_output(['cmake', '--version'])
            except OSError:
                raise RuntimeError("CMake not installed")

            # Check for HDF5 environment variables
            self.hdf5_include_dir = os.environ.get('HDF5_INCLUDE_DIR')
            self.hdf5_lib_dir = os.environ.get('HDF5_LIB_DIR')

            if not self.hdf5_include_dir or not self.hdf5_lib_dir:
                raise RuntimeError("HDF5_INCLUDE_DIR and HDF5_LIB_DIR environment variables must be set")

            
            # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
            ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
            extdir = ext_fullpath.parent.resolve()
            
            debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
            cfg = "Debug" if debug else "Release"
            build_args = ['--config', cfg]
        
            cmake_args = [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
                    f"-DPYTHON_EXECUTABLE={sys.executable}",
                    f"-DCMAKE_BUILD_TYPE={cfg}",
                    f"-DHDF5_INCLUDE_DIR={self.hdf5_include_dir}",
                    f"-DHDF5_LIB_DIR={self.hdf5_lib_dir}"
            ]

            # Adding CMake arguments set as environment variable
            # (needed e.g. to build for ARM OSx on conda-forge)
            if "CMAKE_ARGS" in os.environ:
                cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

            # VERSION_INFO pass version information from python to c++ files
            cmake_args += [f"-DVERSION_INFO={self.distribution.get_version()}"]

            build_temp = Path(self.build_temp) / ext.name
            if not build_temp.exists():
                build_temp.mkdir(parents=True)

            subprocess.run(
                ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
            )
            subprocess.run(
                ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
            )

# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(setuptools.Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


if __name__ == "__main__":
    setuptools.setup(
        name='uf3',
        version='0.4.0',
        description='Ultra-Fast Force Fields for molecular dynamics',
        long_description=long_description,
        url='https://github.com/uf3/uf3',
        author='Stephen R. Xie, Matthias Rupp',
        author_email='sxiexie@ufl.edu',
        license='Apache 2.0',
        packages=setuptools.find_packages(exclude=["tests"]),
        install_requires=install_requires,
        extras_require=extras_require,
        classifiers=[
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'Operating System :: OS Independent',
            'Topic :: Scientific/Engineering'
        ],
        python_requires='>=3.9, <3.13',
        tests_require=test_requires,
        ext_modules=[CMakeExtension('uf3.representation.ultra_fast_featurize',sourcedir='UltraFastFeaturization/cmake/')],
        cmdclass=dict(build_ext=CMakeBuild),
        zip_safe=False
    )

