# Ultra-Fast Force Fields (UF<sup>3</sup>): Interfacing with a VASP Project

## Usage

Add vasprun.xml files and/or entire VASP run directories to ```data``` OR specify "experiment_path" in settings.yaml to point to GASP's garun/temp/ directory.

Note: current implementation parses every ionic step for training data. Consider testing with a smaller number of vaspruns when limited by time/memory constraints. 

Implementation of downselection of ionic steps per run, based on energy differences, is coming soon.

```
python example.py settings.yaml
```
