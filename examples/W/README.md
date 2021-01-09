# Ultra-Fast Force Fields (UF<sup>3</sup>): Tungsten Examples

## Dataset

The Tungsten dataset is available at [qmml.org](https://qmml.org/datasets.html), maintained by Dr. Matthias Rupp.

It was originally available at [libatoms.org](http://www.libatoms.org/), maintained by Dr. Gábor Csányi.

Dataset citation: 
[Accuracy and transferability of Gaussian approximation potential models for tungsten. Wojciech J. Szlachta, Albert P. Bartók, and Gábor Csányi
Phys. Rev. B 90, 104108.](https://doi.org/10.1103/physrevb.90.104108)

## Potentials

Disclaimer: The potentials provided in ```uf3/examples/potentials``` are not intended to yield accurate or transferable results. The potentials in this directory were fit using a training set of 1939 configurations (only 20% of the data). Each entry in ```training_idx.txt``` corresponds to the (n-1)th configuration of the dataset file ```w-14.xyz```.

* The included Lennard-Jones (LJ) and Morse potentials were fit using BFGS algorithm.

* The EAM4 potential is available at [NIST's Interatomic Potential Repository](https://www.ctcms.nist.gov/potentials/entry/2013--Marinica-M-C-Ventelon-L-Gilbert-M-R-et-al--W-4/).

* The included SNAP and qSNAP potentials were fit using [mlearn](https://github.com/materialsvirtuallab/mlearn), which has since been superceded by [maml](https://github.com/materialsvirtuallab/maml).

* The included GAP potential was fit using [QUIP](https://github.com/libAtoms/QUIP), using the [GAP](https://libatoms.github.io/GAP/) plugin.

* The included UF potential was fit using 25 knots using the LAMMPS knot-spacing style.
