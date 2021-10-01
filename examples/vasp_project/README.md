# Ultra-Fast Force Fields (UF<sup>3</sup>): Interfacing with a VASP Project

WIP

## Usage

Add vasprun.xml files, OUTCAR files, and/or entire VASP run directories to ```data``` OR modify "experiment_path" in settings.yaml to point to a project directory e.g. GASP's garun/temp/ directory.

Parsing and Preprocessing (required time: minutes):
```
python preprocess.py settings.yaml
```

Featurization (required time: minutes~hours):
```
python featurize.py settings.yaml
```

Fitting (required time: seconds):
```
python learning.py settings.yaml
```

Testing, Postprocessing, Plotting (required time: seconds)
```
python postprocess.py settings.yaml
```
