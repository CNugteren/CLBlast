CLBlast: Tuning for better performance
================

This document describes how to tune CLBlast for better performance and lists for which devices tuned kernels are already available. For other information about CLBlast, see the [main README](../README.md).


Already tuned-for devices
-------------

The CLBlast library is already tuned for the most commonly used OpenCL devices and it's gradually being extended to other devices as well. For unseen devices CLBlast will make use of common-best tuning values for similar architectures (e.g. AMD Fiji) or in general similar devices (e.g. AMD GPUs), so performance might still be decent. The current release of CLBlast is tuned for the following devices:

* NVIDIA GPUs:
  - GRID K520
  - GeForce GT 650M
  - GeForce GTX 480
  - GeForce GTX 580
  - GeForce GTX 670
  - GeForce GTX 680
  - GeForce GTX 750
  - GeForce GTX 750 Ti
  - GeForce GTX 760 Ti
  - GeForce GTX 920MX
  - GeForce GTX 970
  - GeForce GTX 980
  - GeForce GTX 1070
  - GeForce GTX 1070 Ti
  - GeForce GTX 1080
  - GeForce GTX 1080 Ti
  - GeForce GTX TITAN
  - GeForce GTX TITAN Black
  - GeForce GTX TITAN X
  - TITAN X (Pascal)
  - Tesla K20m
  - Tesla K40m
  - Tesla P100 16GB
* AMD GPUs:
  - Radeon HD 6750M
  - Radeon HD 6770M
  - Radeon HD 7970
  - Radeon R9 270X
  - Radeon R9 290X
  - Radeon R9 M370X
  - Radeon R9 380
  - Radeon RX 480
  - Radeon R9 Fury X
  - Radeon Pro 580
* Intel GPUs:
  - HD Graphics 530
  - HD Graphics 5500 BroadWell U-Processor GT2
  - HD Graphics 6000 BroadWell U-Processor GT3
  - HD Graphics 630
  - HD Graphics Haswell Ultrabook GT2 Mobile
  - HD Graphics IvyBridge M GT2
  - HD Graphics Skylake ULT GT2
  - Iris
  - Iris Pro
* Intel CPUs:
  - Core i5-4570
  - Core i5-4590S
  - Core i5-6200U
  - Core i7-920
  - Core i7-2670QM
  - Core i7-3770K
  - Core i7-4790K
  - Core i7-5930K
  - Core i7-6770HQ
  - Xeon E5-2630 v3
  - Xeon E5-2630 v4
* Other devices:
  - ARM Mali-T628 GPU
  - ARM Mali-T760 GPU
  - Qualcomm Adreno 330 GPU
  - Intel MIC

If your device is not (yet) among this list or if you want to tune CLBlast for specific parameters (e.g. rectangular matrix sizes), you should run the included tuners.


Compiling and running the tuners
-------------

The included CLBlast tuners are compiled with the default CMake options. If they are not compiled, make sure you are specifing `-DTUNERS=ON`, for example as follows:

    cmake -DTUNERS=ON ..

Compiling with `-DTUNERS=ON` will generate a number of tuners, each named `clblast_tuner_xxxxx`, in which `xxxxx` corresponds to a `.opencl` kernel file as found in `src/kernels`. These kernels corresponds to routines (e.g. `xgemm`) or to common pre-processing or post-processing kernels (`copy` and `transpose`). Running such a tuner will test a number of parameter-value combinations on your device and report which one gave the best performance. Running `make alltuners` runs all tuners for all precisions in one go. You can set the default device and platform for `alltuners` by setting the `CLBLAST_DEVICE` and `CLBLAST_PLATFORM` environmental variables. Alternatively, you can also manually run each of the tuners for each of the precisions. Here's an example to tune the `axpy` kernels for 64-bit precision on device 0 of platform 0:

    ./clblast_tuner_xaxpy --precision 64 --device 0 --platform 0

The kernels `gemm` and `gemm_direct` have too many parameters to explore. Therefore, they will run in two stages: a first stage with a fixed limited number of parameter combinations, and a second stage with a random selection from a much larger search space. The random fraction is determined by the `fraction` argument on the command-line.

There are also several routine-level tuners. They tune inter-kernel parameters and should only be run after the kernels are tuned. However, they do automatically pick up kernel tuning results from the current folder if there are any. An example is the GEMM routine tuner, which determines when to use the direct or the in-direct GEMM kernel.

Here are all the tuners included in the `make alltuners` target (in the same order) with all their precision arguments:

    ./clblast_tuner_copy_fast -precision 32
    ./clblast_tuner_copy_fast -precision 64
    ./clblast_tuner_copy_fast -precision 3232
    ./clblast_tuner_copy_fast -precision 6464
    ./clblast_tuner_copy_fast -precision 16
    ./clblast_tuner_copy_pad -precision 32
    ./clblast_tuner_copy_pad -precision 64
    ./clblast_tuner_copy_pad -precision 3232
    ./clblast_tuner_copy_pad -precision 6464
    ./clblast_tuner_copy_pad -precision 16
    ./clblast_tuner_transpose_fast -precision 32
    ./clblast_tuner_transpose_fast -precision 64
    ./clblast_tuner_transpose_fast -precision 3232
    ./clblast_tuner_transpose_fast -precision 6464
    ./clblast_tuner_transpose_fast -precision 16
    ./clblast_tuner_transpose_pad -precision 32
    ./clblast_tuner_transpose_pad -precision 64
    ./clblast_tuner_transpose_pad -precision 3232
    ./clblast_tuner_transpose_pad -precision 6464
    ./clblast_tuner_transpose_pad -precision 16
    ./clblast_tuner_xaxpy -precision 32
    ./clblast_tuner_xaxpy -precision 64
    ./clblast_tuner_xaxpy -precision 3232
    ./clblast_tuner_xaxpy -precision 6464
    ./clblast_tuner_xaxpy -precision 16
    ./clblast_tuner_xdot -precision 32
    ./clblast_tuner_xdot -precision 64
    ./clblast_tuner_xdot -precision 3232
    ./clblast_tuner_xdot -precision 6464
    ./clblast_tuner_xdot -precision 16
    ./clblast_tuner_xger -precision 32
    ./clblast_tuner_xger -precision 64
    ./clblast_tuner_xger -precision 3232
    ./clblast_tuner_xger -precision 6464
    ./clblast_tuner_xger -precision 16
    ./clblast_tuner_xgemm -precision 32
    ./clblast_tuner_xgemm -precision 64
    ./clblast_tuner_xgemm -precision 3232
    ./clblast_tuner_xgemm -precision 6464
    ./clblast_tuner_xgemm -precision 16
    ./clblast_tuner_xgemm_direct -precision 32
    ./clblast_tuner_xgemm_direct -precision 64
    ./clblast_tuner_xgemm_direct -precision 3232
    ./clblast_tuner_xgemm_direct -precision 6464
    ./clblast_tuner_xgemm_direct -precision 16
    ./clblast_tuner_xgemv -precision 32
    ./clblast_tuner_xgemv -precision 64
    ./clblast_tuner_xgemv -precision 3232
    ./clblast_tuner_xgemv -precision 6464
    ./clblast_tuner_xgemv -precision 16
    ./clblast_tuner_invert -precision 32
    ./clblast_tuner_invert -precision 64
    ./clblast_tuner_invert -precision 3232
    ./clblast_tuner_invert -precision 6464
    ./clblast_tuner_invert -precision 16
    ./clblast_tuner_routine_xgemm -precision 32
    ./clblast_tuner_routine_xgemm -precision 64
    ./clblast_tuner_routine_xgemm -precision 3232
    ./clblast_tuner_routine_xgemm -precision 6464
    ./clblast_tuner_routine_xgemm -precision 16
    ./clblast_tuner_routine_xtrsv -precision 32
    ./clblast_tuner_routine_xtrsv -precision 64
    ./clblast_tuner_routine_xtrsv -precision 3232
    ./clblast_tuner_routine_xtrsv -precision 6464
    ./clblast_tuner_routine_xtrsv -precision 16


Using the tuning results
-------------

The tuners output a JSON-file with the results. The best results need to be added to `src/database/kernels/xxxxx.hpp` in the appropriate section. However, this can be done automatically based on the JSON-data using a Python (2.7 or 3.x) script in `scripts/database/database.py`. If you want the found parameters to be included in future releases of CLBlast, please attach the JSON files to the corresponding issue on GitHub or [email the main author](http://www.cedricnugteren.nl).

In summary, tuning the entire library for your device can be done as follows (starting from the root of the CLBlast folder):

    mkdir build
    cd build
    cmake -DTUNERS=ON ..
    make
    make alltuners
    python ../scripts/database/database.py . ..
    make


Tuning using the API (advanced users only)
-------------

Apart from running the tuning binaries, it is also possible to run the tuners programmatically through the CLBlast API. This could be useful if you want to tune for non-standard arguments (e.g. a rectangular or very small matrix). The tuning results can then also be set programmatically using `OverrideParameters`.

The tuning API does not perform any disk or stdout I/O, thus it is not possible to track progress. Running the regular tuner binaries should give an idea of the amount of configurations to explore for a particular device, thus giving an indication of a good value for the `fraction` argument (see the [API documentation](api.md) for more details).


Inspecting and changing tuning parameters at run-time
-------------

Alternatively, you can also supply your tuning parameters programmatically through the CLBlast API. This is especially useful if you tune for specific non-standard arguments (e.g. a rectangular or a very small matrix). To do so, you can call the `OverrideParameters` function which will set new parameters for a specific kernel. At the first next call of the target routine, CLBlast will compile a new binary and use it together with the new parameters from then on. Until `OverrideParameters` is called again of course. This is the API:

    StatusCode PUBLIC_API OverrideParameters(const cl_device_id device, const std::string &kernel_name,
                                             const Precision precision,
                                             const std::unordered_map<std::string,size_t> &parameters)

To inspect current behaviour, you can also retrieve the parameters for a specific device and kernel combination:

    StatusCode PUBLIC_API RetrieveParameters(const cl_device_id device, const std::string &kernel_name,
                                             const Precision precision,
                                             std::unordered_map<std::string,size_t> &parameters)

These two functions require/retrieve the parameters as given in [src/database/kernels](../src/database/kernels), i.e.:

| Kernel name         | Parameters            |
| --------------------|-----------------------|
| Xaxpy               |  VW, WGS, WPT         |
| Xdot                |  WGS1, WGS2           |
| Xgemv               |  WGS1, WPT1           |
| XgemvFast           |  VW2, WGS2, WPT2      |
| XgemvFastRot        |  VW3, WGS3, WPT3      |
| Xger                |  WGS1, WGS2, WPT      |
| Xtrsv               |  TRSV_BLOCK_SIZE      |
| Xgemm               |  GEMMK, KREG, KWG, KWI, MDIMA, MDIMC, MWG, NDIMB, NDIMC, NWG, SA, SB, STRM, STRN, VWM, VWN |
| XgemmDirect         |  KWID, MDIMAD, MDIMCD, NDIMBD, NDIMCD, PADA, PADB, VWMD, VWND, WGD |
| Copy                |  COPY_DIMX, COPY_DIMY, COPY_VW, COPY_WPT |
| Pad                 |  PAD_DIMX, PAD_DIMY, PAD_WPTX, PAD_WPTY |
| Transpose           |  TRA_DIM, TRA_PAD, TRA_SHUFFLE, TRA_WPT |
| Padtranspose        |  PADTRA_PAD, PADTRA_TILE, PADTRA_WPT |
| Invert              |  INTERNAL_BLOCK_SIZE  |
| TrsvRoutine         |  TRSV_BLOCK_SIZE      |


Tuning OpenCL compiler options
-------------

For all of CLBlast's APIs, it is possible to optionally set an OS environmental variable `CLBLAST_BUILD_OPTIONS` to pass specific build options to the OpenCL compiler. Also make sure this is set in the same way when running the tuners.


Which kernels are used for which routines?
-------------

To find out which tuners to run for which routines, you can use the table below. The kernel names correspond to the tuner binaries, the tuner API, and to the arguments for `OverrideParameters` and `RetrieveParameters`.

| Routines                                                                 | Kernel(s) / Tuner(s)            |
| -------------------------------------------------------------------------|---------------------------------|
| AXPY COPY SCAL SWAP OMATCOPY AXPYBATCHED                                 | Xaxpy                           |
| AMAX ASUM DOT DOTC DOTU NRM2 SUM MAX MIN AMIN                            | Xdot                            |
| GBMV GEMV HBMV HEMV HPMV SBMV SPMV SYMV TMBV TPMV TRMV TRSV              | Xgemv                           |
| GER GERC GERU HER HER2 HPR HPR2 SPR SPR2 SYR SYR2                        | Xger                            |
| GEMM HEMM HER2K HERK SYMM SYR2K SYRK TRMM GEMMBATCHED GEMMSTRIDEDBATCHED | Xgemm XgemmDirect Copy Pad Transpose Padtranspose |
| TRSM                                                                     | Xgemm XgemmDirect Copy Pad Transpose Padtranspose Invert |
| IM2COL COL2IM                                                            | Copy                            |
