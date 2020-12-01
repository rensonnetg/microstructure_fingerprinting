# Microstructure fingerprinting
Estimation of white matter microstructural properties from a dictionary of Monte Carlo diffusion MRI fingerprints in Python 3.

## Installing & importing
Download a copy of this repository
```
> git clone https://github.com/rensonnetg/microstructure_fingerprinting.git
```

### Occasional use
If you don't plan to use this library on a regular basis, it may be best to manually import it just when you need it. At the top of your Python script, add
```
import os
import sys
# path to MF source files (the folder containing the actual mf.py, mf_utils.py files, etc.)
path_to_mf = os.path.abspath('path/to/loaded/folder/called/microstructure_fingerprinting/microstructure_fingerprinting')
if path_to_mf not in sys.path:
    sys.path.insert(0, path_to_mf)
import mf
import mf_utils as mfu
```
### Frequent use
For a more frequent use of the library, you may wish to permanently add the package to your current Python environment.

Navigate to the folder where this repository was cloned or downloaded (the folder containing the ```setup.py``` file) and install the package as follows
```
> cd microstructure_fingerprinting
> python setup.py install
```

At the top of your Python scripts, import the library as
```
import microstructure_fingerprinting as mf
import microstructure_fingerprinting.mf_utils as mfu
```

## Usage
### [DIPY](https://dipy.org/)-style interface
```
dictionary_file = 'mf_dictionary.mat'

# Instantiate model:
mf_model = mf.MFModel(dictionary_file)

# Fit to data:
MF_fit = mf_model.fit(DWIfile,  # help(mf_model.fit)
                      maskfile,
                      numfasc,  # all arguments after this MUST be named: argname=argvalue
                      peaks=peaks,
                      bvals=bvalfile,
                      bvecs=bvecfile,
                      csf_mask=csf_mask,
                      ear_mask=ear_mask,
                      verbose=3,
                      parallel=False
                      )
                      
# Save estimated parameter maps as NIfTI files:
outputbasename = 'MF_test'
MF_fit.write_nifti(outputbasename)
```
After initializing the model, in an interactive Python shell type ```help(mf_model.fit)``` for details on the arguments and options of the model fitting method.

### Lower-level interface
The main function of the toolbox is ```solve_exhaustive_posweights```, which you would call by doing ```mfu.solve_exhaustive_posweights(Dictionary, y, dicsizes)``` as per the importation guidelines provided above.

Rotating the dictionary along each fascicle's direction in each voxel is currently supported for multi-shell PGSE protocols and AxCaliber-like PGSE protocols (gradients in xy plane only), for fascicles assumed to have cylindrical symmetry, well described by one principal orientation. In the case of multi-shell protocols, use ```mfu.rotate_atom``` or alternatively ```mfu.interp_PGSE_from_multishell``` which can be made faster by a prior call to ```mfu.init_PGSE_multishell_interp```. For AxCaliber-like, xy-plane protocols use ```mfu.rotate_atom_2Dprotocol```.

Visualizing PGSE signals is possible with ```plot_multi_shell_signal``` and ```plot_signal_2Dprotocol```.

Another useful utility is ```mfu.loadmat``` used to load Matlab-style mat-files, which works in some cases where ```scipy.io.loadmat``` fails.

## References
Rensonnet, G., Scherrer, B., Girard, G., Jankovski, A., Warfield, S.K., Macq, B., Thiran, J.P. and Taquet, M., 2019. Towards microstructure fingerprinting: Estimation of tissue properties from a dictionary of Monte Carlo diffusion MRI simulations. NeuroImage, 184, pp.964-980. https://doi.org/10.1016/j.neuroimage.2018.09.076
