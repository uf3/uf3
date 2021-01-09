#!/usr/bin/env python

import os
import setuptools


module_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(module_dir, 'README.md'), "r") as f:
    long_description = f.read()

install_requires = ['numpy',
                    'scipy',
                    'pandas',
                    'ase>=3.20.1']
test_requires = ['pytest']


if __name__ == "__main__":
    setuptools.setup(
        name='uf3',
        version='0.1',
        description='Ultra-Fast Force Fields for molecular dynamics',
        long_description=long_description,
        url='https://github.com/sxie22/uf3',
        author='Stephen R. Xie, Matthias Rupp',
        author_email='sxiexie@ufl.edu',
        license='Apache 2.0',
        packages=['uf3'],
        install_requires=install_requires,
        classifiers=["Programming Language :: Python :: 3.7",
                     'Development Status :: 3 - Alpha',
                     'Intended Audience :: Science/Research',
                     'Operating System :: OS Independent',
                     'Topic :: Scientific/Engineering'],
        tests_require=test_requires,
    )
