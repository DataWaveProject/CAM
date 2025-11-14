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

### Obtaining `FTorch`

To use PyTorch-based neural nets in CAM, we use [`FTorch`](https://github.com/Cambridge-ICCS/FTorch) which needs to be built on the system before we build CAM.

To install `FTorch` on Derecho follow the instructions in section [`FTorch` on Derecho](#ftorch-on-derecho) below.

> [!NOTE]
> The location of the `FTorch` install will be required later when building CAM.

> [!NOTE]
> If you want to build `FTorch` on another system, please follow the general instructions in the `FTorch`
> [documentation](https://github.com/Cambridge-ICCS/FTorch).

### Obtaining `FTorch`-compatible CIME

This fork of CAM will use an `FTorch`-compatible version of the CIME buildsystem. This is specified under the `[cime]` section
of the `Externals.cfg` file in the main CAM directory.

To set up a CIME to use `FTorch` please follow the instructions in section [`FTorch`-compatible CIME](#ftorch-compatible-cime)
below.

### Checkout externals

To fetch the external components run the following command from within the CAM root directory,

```
./manage_externals/checkout_externals
```

### Creating and running a case

Details on creating a case can be found
[here](https://ncar.github.io/CAM/doc/build/html/CAM6.0_users_guide/building-and-running-cam.html) on the NCAR website.
For this work we are using the following testcase which can be set up by running:
```
cd cime/scripts
./create_newcase --case <path_to_testcase_directory> --compset FMTHIST --res ne30pg3_ne30pg3_mg17 --project XXXXXXX --machine derecho
```
from `CAM/cime/scripts/`.

You can then navigate to the case directory at `<path_to_testcase_directory>`.

### Build CAM with `FTorch`

Before we can run `./case.build` we first need to make some manual changes to the `Makefile` located in
`<path_to_testcase_directory>/Tools/Makefile`. This will allow the CIME build system to locate `FTorch`.

From the test case directory, modify `Tools/Makefile` line 602 to set the environment variable `FTORCH_LIB` to the location of
the `FTorch` library on your system.

If you followed the instructions in section [`FTorch` on Derecho](#ftorch-on-derecho) this will be `$HOME/FTorch/bin/ftorch_intel`.

```make
FTORCH_LIB := </path/to/$HOME/>FTorch/bin/ftorch_intel
```
> [!NOTE]
> You will need to use the full filepath to the FTorch install.
> Do not use `~/` or `$HOME/` as this may not work.

### Setting up case details

We can now run `./case.setup` from within the case directory. Once this has been done then edit the generated `user_nl_cam` in
the case directory as required.

> [!NOTE]
> The following settings are provided as an example. These should be tailored to your particular experiment. For more
> information, please see the descriptions below.

#### ML parameterisation for convection

To run CAM using the NN to predict gravity waves and the physics-based model to _piggyback_, we can set the following settings:
```fortran
gw_convect_dp_ml=.true.
gw_convect_dp_ml_compare=.true.
gw_convect_dp_ml_net_path='/path/to/neural/net'
gw_convect_dp_ml_norms='/path/to/norms'
```

* `gw_convect_dp_ml` (`logical`)

   Whether or not to use the ML scheme for gravity waves produced by deep convection. Default: `.false.`

* `gw_convect_dp_ml_compare` (`logical`)

   Whether or not to run a piggybacking comparison of the ML deep convection gravity waves to the original scheme. Only one
   scheme will be used to advance the simulation as dictated by the value of `gw_convect_dp_ml`. Default: `.false.`

* `gw_convect_dp_ml_net_path`

   Absolute filepath to the deep convection gravity wave neural net used when `gw_convect_dp_ml` is set to `.true.` (`.pt`
   extension).

* `gw_convect_dp_ml_norms`

   Absolute filepath to the deep convection gravity wave normalisation weights (NetCDF) used when `gw_convect_dp_ml` is set to `.true.`.

#### ML parameterisation for non local gravity waves

To run CAM using the non local gravity wave ML model to replace all parameterisations use the following configuration
```fortran
use_gw_nlgw_ann=.true.
use_gw_nlgw_unet=.true.
gw_nlgw_model_path_ann='/path/to/ann-scripted-model.pt'
gw_nlgw_model_path_unet='/path/to/unet-scripted-model.pt'
```

* `use_gw_nlgw_ann` (`logical`)

   Whether or not to use the ANN ML scheme for non local gravity waves. Default: `.false.`

* `gw_nlgw_model_path_ann`

   Absolute filepath to the non local gravity wave neural net used when `use_gw_nlgw` is set to `.true.` (`.pt`
   extension).

* `use_gw_nlgw_unet` (`logical`)

   Whether or not to use the UNET ML scheme for non local gravity waves. Default: `.false.`

* `gw_nlgw_model_path_unet`

   Absolute filepath to the non local gravity wave neural net used when `use_gw_nlgw` is set to `.true.` (`.pt`
   extension).

> [!TIP]
> Consider adding the following to generate output diagnostics of variables as desired.
>
> ```fortran
> fincl<n> = 'MYVAR'
> ```

We can then run `./case.build` from within the case directory to build the model.

The case can be run with `./case.submit` from the case directory.

> [!NOTE]
> By default CESM will place outputs in `$SCRATCH/case/` essential parts of which will be moved to `$SCRATCH/archive/case/` after run completion.
> To leave all output in `$SCRATCH/case/` switch 'short term archiving' off by running `./xmlchange DOUT_S=FALSE` in the
> case directory to change `DOUT_S` from `TRUE` to `FALSE`.

#### Output useful variables

CAM has been configured to output tendencies and fluxes to file. These are named `UTGW_NL`, `VTGW_NL` i.e. tendencies in u and v directions. Similarly, fluxes are named `UFLUX_NL` and `VFLUX_NL`.

For instance, add the following to `user_nl_cam`: 

```
nhtfrq=0,5
mfilt=1,60
avgflag_pertape = 'A','I'

fincl2 = 'U','V','T','UTGW_NL','VTGW_NL','UFLUX_NL','VFLUX_NL'
```

This outputs to two history files, 0 and 1, where `fincl1` controls fields in h0 and `fincl2` controls fields in h1. In h0, contains the default variables or fields. 

We output to h1 a reduced set at a higher frequency. In this case, the selected variables are written every 5 timesteps, and then output to disk once 60 time samples have been buffered. See documentation on customising CAM output [here](https://ncar.github.io/CESM-Tutorial/notebooks/namelist/output/output_cam.html). In the above instance files will be output in the following format `<job-name>.cam.h1.1979-01-01-[00000,21600,43200...].nc`.

#### Long runs

Long runs can be achieved by following the documentation [here](https://ncar.github.io/CESM-Tutorial/notebooks/modifications/xml/run_length/restarting.html).

The below `user_nl_cam` content sets up a long run that automatically resubmits. Of course, resubmission can be performed manually by removing the `RESUBMIT` option and toggling `CONTINUE_RUN` to `True` after the first run completes. 

```
! Restart behaviour
CONTINUE_RUN = FALSE   ! First run starts from initial conditions.
                       ! CESM RESUBMIT machinery will auto-set TRUE for follow-up jobs.
                       ! Set to TRUE only for manual restarts after a crash/timeout.

! Automatically resubmit 
RESUBMIT = 9 ! Automatically resubmit 9 times (5 years total)

REST_DATE: -999 ! Don’t force any special date; just use REST_OPTION/REST_N
REST_OPTION: nmonths 
REST_N: 3

STOP_OPTION: nmonths
STOP_N: 6 ! Each job runs for 6 months only to provide plenty of buffer in 12 hour job window

! Sensible output frequencies
nhtfrq=0,-336 ! Every two weeks
mfilt=1,1
avgflag_pertape = 'A','I'

fincl2 = 'U','V','T','UTGW_NL','VTGW_NL','UFLUX_NL','VFLUX_NL'
```


## NOTE: This is **unsupported** development code and is subject to the [CESM developer's agreement](http://www.cgd.ucar.edu/cseg/development-code.html).

### CAM Documentation - https://ncar.github.io/CAM/doc/build/html/index.html

### CAM6 namelist settings - http://www.cesm.ucar.edu/models/cesm2/settings/current/cam_nml.html

Please see the [wiki](https://github.com/ESCOMP/CAM/wiki) for complete documentation on CAM, getting started with git and how to contribute to CAM's development.

### `FTorch` on Derecho

The following steps can be followed to ensure `FTorch` is built to be consistent
with CAM on Derecho.

#### load CAM environment

For compatibility with the version of CAM we are using (branched from the `cam6_3_139` tag) we need to be specific about the environment and compilers we load. The following sequence of modules
are required to build `FTorch` compatible with the intel build of CAM on Derecho:

```bash
module purge
module load ncarenv/23.06
module load intel-oneapi/2023.0.0
module load mkl
module load cmake
module load cuda/11.7.1
```

> [!NOTE]
> In future builds or releases, or on different machines, the environment for building CAM may change. In this case the `FTorch`
> environment should be updated accordingly.

#### obtain Ftorch source

```bash
cd $HOME
git clone git@github.com:Cambridge-ICCS/FTorch.git
cd $HOME/FTorch/src
```

#### build Ftorch

`FTorch` can then be built and installed from `$HOME/Ftorch/src` as described in the documentation with:

```bash
mkdir -p build && cd build
cmake  .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_Fortran_COMPILER=ifort \
  -DCMAKE_C_COMPILER=icc \
  -DCMAKE_CXX_COMPILER=icpc \
  -DCMAKE_PREFIX_PATH=/glade/u/apps/opt/libtorch/2.1.2 \
  -DCMAKE_INSTALL_PREFIX=../../bin/ftorch_intel/
cmake --build . --target install
```

This will build `FTorch` and install it to `$HOME/Ftorch/bin/ftorch_intel`.

### `FTorch`-compatible CIME

We need to use a version of the CIME build system that is capable of linking
our code to `FTorch` when building CAM.

To do this we have modified the `Externals.cfg` file in the main CAM directory to
replace the CIME entry with:

```
[cime]
branch = ftorch_gw
protocol = git
repo_url = https://github.com/Cambridge-ICCS/cime_je
local_path = cime
required = True
```

which points to the [ICCS fork](https://github.com/Cambridge-ICCS/cime_je) of CIME
that allows components to be built with `FTorch`.

Specifically it points to a branch based off of the `cime6.0.175` tag that is compatible
with the latest version of CIME used with this version of CAM (this is the cime tag 
associated with the `cam6_3_139` tag of CAM).
