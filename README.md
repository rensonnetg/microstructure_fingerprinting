# microstructure_fingerprinting
Estimation of white matter microstructural properties from a dictionary of Monte Carlo diffusion fingerprints

## Import
At the top of your Python script, add
```
import sys
mf_path = 'relative/or/absolute/path/to/microstructure_fingerprinting'
if mf_path not in sys.path:
     sys.path.insert(0, mf_path)
import mf_utils as mf
```

## Usage
The main function of the toolbox is ```solve_exhaustive_posweights```, which in the above example you would call by doing ```mf.solve_exhaustive_posweights(Dictionary,   y, dicsizes)``` for instance.
Other important funcions are ```rotate_atom``` used to rotate the dictionary in each voxel and ```loadmat``` used to load Matlab-style mat-files, which works in some cases where ```loadmat``` from the ```scipy.io``` package fails.
