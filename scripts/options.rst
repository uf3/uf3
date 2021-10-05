Ultra-Fast Force Fields (UF3) - YAML/CLI Guide
===================================================

General Settings
----------------

``outputs_path`` (string): directory in which to save outputs such as
trained models, LAMMPS potentials, and plots. 

- Default: *“./outputs”*

``element_list`` (list): elements in the chemical system 

- Example:

::

    element_list:
     - Ca
     - Ti
     - O

``degree`` (integer): maximum degree of atomic interactions to consider.

- Options: *{2, 3}* 

- Default: *2*

``seed`` (integer): seed for random number routines in Numpy. Improves
reproducibility. 

- Default: *0*

Data
----

``db_path`` (string): filename for cached data. Created during
preprocessing. Used during featurization, learning, and postprocessing.

- Default: *“data.db”*

``max_per_file`` (integer): maximum number of samples to take per file.
Farthest-point-sampling is used to select subsets of samples. Typically,
when sampling from a relaxation path, more samples are taken from the
beginning than the end. *-1* takes all entries without farthest-point-sampling.

- Default: *-1*

``min_diff`` (float): minimum energy difference in eV between any two
samples selected with farthest-point-sampling. This avoids oversampling
configurations that are very similar in energy, e.g. at the end of a
relaxation path. 

- Default: *1e-8*

``generate_stats`` (boolean): whether to analyze and summarize pair
interaction distances, observed peaks, and more. 

- Default: *True*

``progress`` (string): style of printing progress. *bar* enables tqdm
progress bars while *text* yields timestamped updates. *None* or *False*
disables progress indicators.
 
- Options: *{“bar”, “text”, None}* 

- Default: *“bar”*

``vasp_pressure`` (boolean): enable correction for external pressure,
subtracting *pressure * volume* term from parsed energies. External
pressure tag *PSTRESS* is extracted from file (INCAR, OUTCAR,
vasprun.xml) inside the same directory. 

- Default: *True*

sources
~~~~~~~

``path`` (string): path to highest-level directory in which to search
for files. 

- Default: *“./data”*

``pattern`` (string): glob pattern for recursive search in *path*. 

- Default: "*"

keys
~~~~

``atoms_key`` (string): column name for atomic configurations in
DataFrame. 

- Default: *“geometry”*

``energy_key`` (string): keyword for energies in data parsing and
``ase.Atoms.info``. 

- Default: *“energy”*

``force_key`` (string): keyword for forces in data parsing and
``ase.Atoms.arrays``. 

- Default: *“forces”*

``size_key`` (string): column name for number of atoms per atomic
configurations in DataFrame. 

- Default: *“size”*

Basis
-----

``r_min`` (dictionary): minimum pair distance per interaction, in
angstroms, to consider for featurization. 

- Note: this value should be
low enough to account for the smallest pair distances expected to appear
in simulations. Otherwise, LAMMPS will fail as soon as two atoms get too
close together. 

- Default: *1.0* for 2B interactions, *[1.0, 1.0, 1.0]*
for 3B interactions

``r_max`` (dictionary): maximum pair distance per interaction, in
angstroms, to consider for featurization. 

- Note: increasing this value
necessarily increases the number of neighbors to consider during
featurization, which increases the computational cost of featurization.

- Default: *6.0* for 2B interactions, *[6.0, 6.0, 6.0]* for 3B
interactions

``resolution`` (dictionary): number of knot intervals per interaction. 

- Note: Due to local support, featurization time does not scale with this
value. However, memory requirements do. In the case of 3B interactions,
the scaling is cubic. 

- Default: *25* for 2B interactions, *[10, 10, 20]* for 3B interactions

``fit_offsets`` (boolean): enable fitting 1-body energies per element,
a.k.a. reference energy or isolated-atom energy. 

- Default: *True*

``trailing_trim`` (integer): force a number of trailing basis functions
for each pair potential to zero during training. Note: if the upper
cutoff distances given in ``r_max`` are too low, then this scheme may
slightly increase error. 

- Default: *3* 

    * ``= 0``: hard cutoff at r_max`` 

    * ``= 1``: function goes to zero at ``r_max`` 

    * ``= 2``: first derivative goes to zero at ``r_max`` 

    * ``= 3``: second derivative goes to zero at ``r_max``

``mask_trim`` (boolean): whether to mask all trimmed basis functions
when caching features. For large values of ``resolution``, this option
greatly reduces filesize. 

- Default: *True*

``knot_strategy`` (string): spacing scheme for placing knots, given a
fixed resolution. When ``read_knots`` is *True* and ``knots_file`` is
provided, this setting is ignored. 

- Options: *{“linear”, “lammps”, “geometric”, “inverse”, }* 

    * ``= linear``: uniform spacing of knots.

    * ``= lammps``: LAMMPS-style r^2 spacing, resulting in higher resolution at longer distances and lower resolution and smaller distances. 

    * ``= geometric``: log(r) spacing, yielding higher resolution at smaller distances.

    * ``= inverse``: 1/r spacing, yielding higher resolution at smaller distances.

- Default: *“linear”*

``knots_path`` (string): filename for knots. If specified, enables
writing and reading of knot sequences.

- Note: if specified and ```load_knots``` is on, any
settings (per interaction) specified in ``r_min``, ``r_max``, and
``resolution`` are ignored. 

- Default: *“knots.json”*

``load_knots`` (boolean): If enabled, read knot sequences from ``knots_path``.

- Default: *False*

``dump_knots`` (boolean): If enabled, write knot sequences to ``knots_path``, overwriting existing files.

- Default: *False*

Features
--------

``db_path`` (string): filename for cached data. Created during
preprocessing. Used during featurization and learning. 

- Default: *“data.db”*

``features_path`` (string): filename for cached features. Created during
featurization. Used during learning. 

- Default: *“features.h5”*

``n_cores`` (integer): maximum number of parallel processes for
featurization. 

- Default: *4*

``parallel`` (string): backend for parallel execution. 

- Options:
*{“python”, “dask”}* 

- Default: *“python”*

Model
-----

``model_path`` (string): filename for serialized model. Created during
learning. Used during prediction. 

- Default: *“model.json”*

Learning
--------

``features_path`` (string): filename for cached features. Created during
featurization. Used during learning. 

- Default: *“features.h5”*

``splits_path`` (string): filename for cached
training-testing-validation and, optionally, cross-validation splits.

- Default: *“splits.json”*

``holdout_split`` (integer, float): number of samples (integer) or
fraction of total samples (float < 1) to partition for holdout. The
remainder is used for training. 

- Default: *0.2*

``cv_split`` (integer): number of partitions to create out of
non-holdout data for cross-validation purposes. 

- Default: *5*

``weight`` (float): weighting parameter for error in energies and forces
during training. Lower values emphasize forces while higher values
emphasize energies. *0.0* disables energy contributions to the fit while
*1.0* disables force contributions to the fit. 

- Default: *0.5*

``regularizer`` (dictionary): ridge and curvature regularization
strengths for 1-body, 2-body, and 3-body interactions. 

- Default:

::

    - ridge_1b: 1e-8     
    
    - ridge_2b: 0     
    
    - ridge_3b: 0     
    
    - curvature_2b: 1e-8    
    
    - curvature_3b: 1e-8
