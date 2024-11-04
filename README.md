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
git checkout coupling
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
FTORCH_LIB := $HOME/FTorch/bin/ftorch_intel
```

### Setting up case details

We can now run `./case.setup` from within the case directory. Once this has been done then edit the generated `user_nl_cam` in
the case directory as required. Add the following lines:

```fortran
gw_convect_dp_ml=.false.
gw_convect_dp_ml_compare=.true.
gw_convect_dp_ml_net_path='/path/to/neural/net'
gw_convect_dp_ml_norms='/path/to/norms'
```

* `gw_convect_dp_ml` (`logical`)

   Whether or not to use the ML scheme for gravity waves produced by deep convection. Default: `.false.`

* `gw_convect_dp_ml_compare` (`logical`)

   Whether or not to run a piggybacking comparison of the ML deep convection gravity waves to the original scheme. Only one
   scheme will be used to advance the simulation as dictated by `gw_convect_deep_ml`. Default: `.false.`

* `gw_convect_dp_ml_net_path`

   Absolute filepath to the deep convection gravity wave neural net used when `gw_convect_dp_ml` is set to `.true.` (`.pt`
   extension).

* `gw_convect_dp_ml_norms`

   Absolute filepath to the deep convection gravity wave normalisation weights (NetCDF) used when `gw_convect_dp_ml` is set to `.true.`.

> [!TIP]
> Consider adding the following to generate output diagnostics of variables as desired.
>
> ```fortran
> fincl<n> = 'MYVAR'
> ```

We can then run `./case.build` from within the case directory to build the model.

The case can be run with `./case.submit` from the case directory.

> [!NOTE]
> By default CESM will place output in `$SCRATCH/case/` and logs/restart files in `$SCRATCH/archive/case/`.
> To place all output with logs in `archive/case` switch 'short term archiving' on by running `./xmlchange DOUT_S=FALSE` in the
> case directory to change `DOUT_S` from `TRUE` to `FALSE`.

## NOTE: This is **unsupported** development code and is subject to the [CESM developer's agreement](http://www.cgd.ucar.edu/cseg/development-code.html).

### CAM Documentation - https://ncar.github.io/CAM/doc/build/html/index.html

### CAM6 namelist settings - http://www.cesm.ucar.edu/models/cesm2/settings/current/cam_nml.html

Please see the [wiki](https://github.com/ESCOMP/CAM/wiki) for complete documentation on CAM, getting started with git and how to contribute to CAM's development.

### `FTorch` on Derecho

The following steps can be followed to ensure `FTorch` is built to be consistent
with CAM on Derecho.

#### load CAM environment

For compatibility with CAM we need to be specific about the environment and compilers we load. The following sequence of modules
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

> [!NOTE]
> When building `FTorch` with `cmake` in the next step, we use the absolute path to `libtorch`. Previously, we loaded the
> `libtorch` module i.e., `module load libtorch/2.1.2` but this conflicts with `ncarenv/23.06`. We do not actually need to load
> the `libtorch` module to use the absolute path, so for now we can ignore this.

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
