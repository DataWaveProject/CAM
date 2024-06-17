# CAM: The Community Atmosphere Model

This fork of CAM is used for the development of machine-learnt gravity wave parameterisations
developed as part of the DataWave project.

The main working branch for this project is `datawave_ml` which was originally branched
from the `cam6_3_139` tag.


## Using this model

### Obtaining CAM

Clone a copy of this repository from git and ensure you checkout the `datawave_ml`
branch on which this work is based:
```
git clone https://github.com/DataWaveProject/CAM.git
cd CAM
git checkout datawave_ml
```
This branch is built upon the `cam6_3_139` tag from the
[main ESMCOMP/CAM repository](https://github.com/ESCOMP/CAM).

### Preparing FTorch and setting the CIME component in CESM

We also need to build and link _FTorch_ which will allow us to use our PyTorch-based
neural nets in CAM. This has two steps.

#### Building FTorch

You need to build _FTorch_ locally on the system following the instructions
[in the documentation](https://github.com/Cambridge-ICCS/FTorch).

Note that if building on Derecho `libtorch` should be loaded using
```
module load libtorch/2.1.2
```
and used to build _FTorch_.\
Further, for compatibility with CAM we need to be specific about the compilers we
load. The following sequence of modules are required to build compatible FTorch on
Derecho:
```
module load ncarenv-basic/23.06
module load ncarenv/23.06
module load intel/2023.0.0
module load cmake
module load cuda/11.7.1
```
FTorch can then be built and installed from `/path/to/ftorch/src/build/` as described in the
documentation with:
```
cmake  .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_Fortran_COMPILER=ifort \
  -DCMAKE_C_COMPILER=icc \
  -DCMAKE_CXX_COMPILER=icpc \
  -DCMAKE_PREFIX_PATH=/glade/u/apps/opt/libtorch/2.1.2 \
  -DCMAKE_INSTALL_PREFIX=../../bin/ftorch_intel/

cmake --build . --target install
```
This will build FTorch and install it to `/path/to/ftorch/bin/ftorch_intel`.

#### Obtaining _FTorch_-compatible CIME

We also need to use a version of the CIME build system that is capable of linking
our code to FTorch.

To do this modify the `Externals.cfg` file in the main CAM directory to
replace the CIME entry with:
```
[cime]
branch = ftorch_gw
protocol = git
repo_url = https://github.com/Cambridge-ICCS/cime_je
local_path = cime
required = True
```
which is the [ICCS fork](https://github.com/Cambridge-ICCS/cime_je) of CIME
that allows components to be built with FTorch.

### Checkout externals

You can now run, from within the CAM root directory,
```
./manage_externals/checkout_externals
```
to fetch the external components.

Once this is complete remember to set the location of FTorch in the CIME Makefile
as [described above](#preparing-ftorch-and-setting-the-cime-component-in-cesm)

### Creating and running a case

Details on creating a case can be found
[here](https://ncar.github.io/CAM/doc/build/html/CAM6.0_users_guide/building-and-running-cam.html) on the NCAR website.
For this work we are using the following testcase which can be set up by running:
```
./create_newcase --case <path_to_testcase_directory> --compset FMTHIST --res ne30pg3_ne30pg3_mg17 --project XXXXXXX --machine derecho
```
from `CAM/cime/scripts/`.

You can then navigate to `<path_to_testcase_directory>`.
To couple to FTorch modify `Tools/Makefile` line 600 to set the environment variable
FTORCH_LIB to the location of the FTorch library on your system (defined above as
`/path/to/ftorch/bin/ftorch_intel`).

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
