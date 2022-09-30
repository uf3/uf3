#!/usr/bin/env python

import os
import setuptools


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

if __name__ == "__main__":
    setuptools.setup(
        name='uf3',
        version='0.3.2',
        description='Ultra-Fast Force Fields for molecular dynamics',
        long_description=long_description,
        url='https://github.com/uf3/uf3',
        author='Stephen R. Xie, Matthias Rupp',
        author_email='sxiexie@ufl.edu',
        license='Apache 2.0',
        packages=setuptools.find_packages(exclude=["tests"]),
        install_requires=install_requires,
        extras_require=extras_require,
        classifiers=["Programming Language :: Python :: 3.7",
                     'Development Status :: 3 - Alpha',
                     'Intended Audience :: Science/Research',
                     'Operating System :: OS Independent',
                     'Topic :: Scientific/Engineering'],
        tests_require=test_requires,
    )