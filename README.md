# CAM: The Community Atmosphere Model

This fork of CAM is used for the development of machine-learnt gravity wave parameterisations
developed as part of the DataWave project.

The main working branch for this project is `datawave_ml` which was originally branched
from the `cam6_3_139` tag.


## Using this model in a CESM Run

### Obtaining CESM

Clone a copy of CESM from git and checkout the `cesm2.1.5` tag on which this work is based:
```
git clone https://github.com/escomp/cesm.git my_cesm_sandbox_2_1
cd my_cesm_sandbox_2_1/
git checkout cesm2.1.5
```

### Setting CAM version in CESM

To use this model in a CESM run you need to modify the `Externals.cfg` file in the
main CESM directory to replace the CAM entry with:
```
[cam]
branch = datawave_ml
protocol = git
repo_url = https://github.com/DataWaveProject/CAM
local_path = components/cam
externals = Externals_CAM.cfg
required = True
```
This will pull the `datawave_cam` branch of this repo in as the CAM component.

### Preparing FTorch and setting the CIME component in CESM

We also need to build and link _FTorch_ which will allow us to use our PyTorch-based
neural nets in CAM/CESM. This has two steps.

First modify again the `Externals.cfg` file in the main CESM directory to
replace the CIME entry with:
```
[cime]
branch = ftorch_forpy_cime
protocol = git
repo_url = https://github.com/Cambridge-ICCS/cime_je
local_path = cime
required = True
```
which is the [ICCS fork](https://github.com/Cambridge-ICCS/cime_je) of CIME
that allows CESM to be built with FTorch.

In addition to this you also need to build _FTorch_ locally on the system
following the instructions
[in the documentation](https://github.com/Cambridge-ICCS/FTorch).\
Note that if building on Derecho `libtorch` should be loaded using
```
module load libtorch/2.1.2
```
and used to build _FTorch_.

After checking out the externals ([see below](#checkout-externals)) you will need to
modify `cime/scripts/Tools/Makefile` line 567 to set the environment variable FTORCH_LOC to the location of the FTorch library on your system

### Checkout externals

You can now run, from within the CESM root directory,
```
./manage_externals/checkout_externals
```
to fetch the external components.

Once this is complete remember to set the location of FTorch in the CIME Makefile
as [described above](#preparing-ftorch-and-setting-the-cime-component-in-cesm)

### Creating and running a case

Details on creating a case can be found
[here](https://ncar.github.io/CAM/doc/build/html/CAM6.0_users_guide/building-and-running-cam.html) on the NCAR website.
For this work we are using the gate III testcase which can be set up by running:
```
./create_newcase --case <path_to_testcase_directory> --compset FMTHIST --res ne30pg3_ne30pg3_mg17 --project XXXXXXX --machine derecho
```
from `<cesm_root>/cime/scripts/`.

Once this has been done then edit `user_nl_cam` in the case directory as required.
This is a CAM namelist generated from the default for the case.
Add the following lines:

#. `gw_convect_dp_ml = 'on'`\
    This is the switch to use our new ML convective-gw scheme instead of the default.\
    Other options are `'off'` (default - use original), `'bothoff'` (run both schemes
    but use default for simulation), and `'bothon'` (run both but use ML for simulation).
#. `convect_dp_ml_model = '<PATH/TO/MODEL.pt>'`\
    The path to the your saved PyTorch model.

Also consider adding:
```
fincl<n> = 'MYVAR'
```
to generate output diagnostics of variables as desired.

We can then run `./case.setup` and `./case.build` from within the case directory.

**Note:**\
By default CESM will place output in `/glade/scratch/user/case/`
and logs/restart files in `/glade/scratch/user/archive/case/`.
To place all output with logs in `archive/case` switch 'short term archiving' on by
editing `env_run.xml` in the case directory to change `DOUT_S` from `FALSE` to `TRUE`.

## NOTE: This is **unsupported** development code and is subject to the [CESM developer's agreement](http://www.cgd.ucar.edu/cseg/development-code.html).

### CAM Documentation - https://ncar.github.io/CAM/doc/build/html/index.html

### CAM6 namelist settings - http://www.cesm.ucar.edu/models/cesm2/settings/current/cam_nml.html

Please see the [wiki](https://github.com/ESCOMP/CAM/wiki) for complete documentation on CAM, getting started with git and how to contribute to CAM's development.
