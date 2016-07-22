
CLBlast: The tuned OpenCL BLAS library
================

| | master | development |
|-----|-----|-----|
| Linux/OS X | [![Build Status](https://travis-ci.org/CNugteren/CLBlast.svg?branch=master)](https://travis-ci.org/CNugteren/CLBlast/branches) | [![Build Status](https://travis-ci.org/CNugteren/CLBlast.svg?branch=development)](https://travis-ci.org/CNugteren/CLBlast/branches) |
| Windows | [![Build Status](https://ci.appveyor.com/api/projects/status/github/cnugteren/clblast?branch=master&svg=true)](https://ci.appveyor.com/project/CNugteren/clblast) | [![Build Status](https://ci.appveyor.com/api/projects/status/github/cnugteren/clblast?branch=development&svg=true)](https://ci.appveyor.com/project/CNugteren/clblast) |

CLBlast is a modern, lightweight, performant and tunable OpenCL BLAS library written in C++11. It is designed to leverage the full performance potential of a wide variety of OpenCL devices from different vendors, including desktop and laptop GPUs, embedded GPUs, and other accelerators. CLBlast implements BLAS routines: basic linear algebra subprograms operating on vectors and matrices.

This preview-version is not yet tuned for all OpenCL devices: __out-of-the-box performance on some devices might be poor__. See below for a list of already tuned devices and instructions on how to tune yourself and contribute to future releases of the CLBlast library.


Why CLBlast and not clBLAS or cuBLAS?
-------------

Use CLBlast instead of clBLAS:

* When you care about achieving maximum performance.
* When you want to be able to inspect the BLAS kernels or easily customize them to your needs.
* When you run on exotic OpenCL devices for which you need to tune yourself.
* When you are still running on OpenCL 1.1 hardware.
* When you value an organized and modern C++ codebase.
* When you target Intel CPUs and GPUs or embedded devices
* When you can benefit from the increased performance of half-precision fp16 data-types.

Use CLBlast instead of cuBLAS:

* When you want your code to run on devices other than NVIDIA CUDA-enabled GPUs.
* When you want to tune for a specific configuration (e.g. rectangular matrix-sizes).
* When you sleep better if you know that the library you use is open-source.
* When you are using OpenCL rather than CUDA.

When not to use CLBlast:

* When you run on NVIDIA's CUDA-enabled GPUs only and can benefit from cuBLAS's assembly-level tuned kernels.


Compilation and installation
-------------

The pre-requisites for compilation of CLBlast are:

* CMake version 2.8.10 or higher
* A C++11 compiler, for example:
  - GCC 4.7.0 or newer
  - Clang 3.3 or newer
  - AppleClang 5.0 or newer
  - ICC 14.0 or newer
  - MSVC (Visual Studio) 2015 or newer
* An OpenCL 1.1 or newer library, for example:
  - Apple OpenCL
  - NVIDIA CUDA SDK
  - AMD APP SDK
  - Intel OpenCL
  - Beignet

An example of an out-of-source build using a command-line compiler and make (starting from the root of the CLBlast folder):

    mkdir build
    cd build
    cmake ..
    make
    sudo make install

When using Visual Studio, the project-files can be generated as follows:

    mkdir build
    cd build
    cmake -G "Visual Studio 14 Win64" ..

A custom installation folder can be specified when calling CMake:

    cmake -DCMAKE_INSTALL_PREFIX=/path/to/install/directory ..


Using the library
-------------

Like clBLAS and cuBLAS, CLBlast also requires OpenCL device buffers as arguments to its routines. This means you'll have full control over the OpenCL buffers and the host-device memory transfers. CLBlast's API is designed to resemble clBLAS's C API as much as possible, requiring little integration effort in case clBLAS was previously used. Using CLBlast starts by including the C++ header:

    #include <clblast.h>

Or alternatively the plain C version:

    #include <clblast_c.h>

Afterwards, any of CLBlast's routines can be called directly: there is no need to initialize the library. The available routines and the required arguments are described in the above mentioned include files and the included [API documentation](doc/clblast.md). Additionally, a couple of stand-alone example programs are included in the `samples` subfolder. They can optionally be compiled using the CMake infrastructure of CLBlast by providing the `-DSAMPLES=ON` flag, for example as follows:

    cmake -DSAMPLES=ON ..


Using the tuners (optional)
-------------

The CLBlast library will be tuned in the future for the most commonly used OpenCL devices. This pre-release of CLBlast is only tuned for a limited number of devices, in particular those with the following `CL_DEVICE_NAME` values:

* NVIDIA GPUs:
  - GRID K520
  - GeForce GTX 480
  - GeForce GTX 670
  - GeForce GTX 680
  - GeForce GTX 750
  - GeForce GTX 750 Ti
  - GeForce GTX 980
  - GeForce GTX 1070
  - GeForce GTX Titan
  - GeForce GTX Titan X
  - Tesla K20m
  - Tesla K40m
* AMD GPUs:
  - AMD Radeon R9 M370X Compute Engine
  - Hawaii
  - Oland
  - Pitcairn
  - Tahiti
* Intel GPUs:
  - HD Graphics 530
  - HD Graphics Haswell Ultrabook GT2 Mobile
  - HD Graphics Skylake ULT GT2
  - Iris
  - Iris Pro
* Intel CPUs:
  - Core i5-6200U
  - Core i7-3770K
  - Core i7-5930K
* Other devices:
  - ARM Mali-T628 GPU
  - Intel MIC

If your device is not (yet) among this list or if you want to tune CLBlast for specific parameters (e.g. rectangular matrix sizes), you should compile the library with the optional tuners by specifing `-DTUNERS=ON`, for example as follows:

    cmake -DTUNERS=ON ..

Note that CLBlast's tuners are based on the [CLTune auto-tuning library](https://github.com/CNugteren/CLTune), which has to be installed separately (requires version 2.3.1 or higher).

Compiling with `-DTUNERS=ON` will generate a number of tuners, each named `clblast_tuner_xxxxx`, in which `xxxxx` corresponds to a `.opencl` kernel file as found in `src/kernels`. These kernels corresponds to routines (e.g. `xgemm`) or to common pre-processing or post-processing kernels (`copy` and `transpose`). Running such a tuner will test a number of parameter-value combinations on your device and report which one gave the best performance. Running `make alltuners` runs all tuners for all precisions in one go. You can set the default device and platform for `alltuners` by setting the `DEFAULT_DEVICE` and `DEFAULT_PLATFORM` environmental variables before running CMake.

The tuners output a JSON-file with the results. The best results need to be added to `src/database/kernels/xxxxx.hpp` in the appropriate section. However, this can be done automatically based on the JSON-data using a Python script in `scripts/database/database.py`. If you want the found parameters to be included in future releases of CLBlast, please attach the JSON files to the corresponding issue on GitHub or [email the main author](http://www.cedricnugteren.nl).

In summary, tuning the entire library for your device can be done as follows (starting from the root of the CLBlast folder):

    mkdir build
    cd build
    cmake -DTUNERS=ON ..
    make
    make alltuners
    python ../scripts/database/database.py . ..
    make


Compiling the correctness tests (optional)
-------------

To make sure CLBlast is working correctly on your device (recommended), compile with the tests enabled by specifying `-DTESTS=ON`, for example as follows:

    cmake -DTESTS=ON ..

To build these tests, another BLAS library is needed to serve as a reference. This can be either:

* The OpenCL BLAS library [clBLAS](http://github.com/clMathLibraries/clBLAS) (maintained by AMD)
* A regular CPU Netlib BLAS library, e.g.:
  - OpenBLAS
  - BLIS
  - Accelerate

Afterwards, executables in the form of `clblast_test_xxxxx` are available, in which `xxxxx` is the name of a routine (e.g. `xgemm`). Note that CLBlast is tested for correctness against [clBLAS](http://github.com/clMathLibraries/clBLAS) and/or a regular CPU BLAS library. If both are installed on your system, setting the command-line option `-clblas 1` or `-cblas 1` will select the library to test against for the `clblast_test_xxxxx` executables. All tests have a `-verbose` option to enable additional diagnostic output. They also have a `-full_test` option to increase coverage further.

All tests can be run directly together in one go through the `make alltests` target or using CTest (`make test` or `ctest`). In the latter case the output is less verbose. Both cases allow you to set the default device and platform to non-zero by setting the `DEFAULT_DEVICE` and `DEFAULT_PLATFORM` environmental variables before running CMake.


Compiling the performance tests/clients (optional)
-------------

To test the performance of CLBlast and compare optionally against [clBLAS](http://github.com/clMathLibraries/clBLAS) or a CPU BLAS library (see above for requirements), compile with the clients enabled by specifying `-DCLIENTS=ON`, for example as follows:

    cmake -DCLIENTS=ON ..

The performance tests come in the form of client executables named `clblast_client_xxxxx`, in which `xxxxx` is the name of a routine (e.g. `xgemm`). These clients take a bunch of configuration options and directly run CLBlast in a head-to-head performance test against optionally clBLAS and/or a CPU BLAS library. You can use the command-line options `-clblas 1` or `-cblas 1` to select a library to test against.

The folder `doc/performance` contains some PDF files with performance results on tested devices. Performance is compared in this case against a tuned version of the clBLAS library. These graphs can be generated automatically on your own device. First, compile CLBlast with the clients enabled. Then, make sure your installation of the reference clBLAS is performance-tuned by running the `tune` executable. Finally, run one of the graph-scripts found in `scripts/graphs` using R. For example, to generate the Xgemm PDF on device 1 of platform 0 from the `build` subdirectory:

    Rscript ../scripts/graphs/xgemm.r 0 1

Note that the CLBlast library provides pre-tuned parameter-values for some devices only: if your device is not among these, then out-of-the-box performance might be poor. See above under `Using the tuners` to find out how to tune for your device.

In case performance is still sub-optimal or something else is wrong, CLBlast can be build in verbose mode for (performance) debugging by specifying `-DVERBOSE=ON` to CMake.


Supported routines
-------------

CLBlast supports almost all the Netlib BLAS routines plus a couple of extra non-BLAS routines. The supported BLAS routines are marked with '✔' in the following tables. Routines marked with '-' do not exist: they are not part of BLAS at all. The different data-types supported by the library are:

* __S:__ Single-precision 32-bit floating-point (`float`).
* __D:__ Double-precision 64-bit floating-point (`double`).
* __C:__ Complex single-precision 2x32-bit floating-point (`std::complex<float>`).
* __Z:__ Complex double-precision 2x64-bit floating-point (`std::complex<double>`).
* __H:__ Half-precision 16-bit floating-point (`cl_half`). See section 'Half precision' for more information.

| Level-1  | S | D | C | Z | H |
| ---------|---|---|---|---|---|
| xSWAP    | ✔ | ✔ | ✔ | ✔ | ✔ |
| xSCAL    | ✔ | ✔ | ✔ | ✔ | ✔ |
| xCOPY    | ✔ | ✔ | ✔ | ✔ | ✔ |
| xAXPY    | ✔ | ✔ | ✔ | ✔ | ✔ |
| xDOT     | ✔ | ✔ | - | - | ✔ |
| xDOTU    | - | - | ✔ | ✔ | - |
| xDOTC    | - | - | ✔ | ✔ | - |
| xNRM2    | ✔ | ✔ | ✔ | ✔ | ✔ |
| xASUM    | ✔ | ✔ | ✔ | ✔ | ✔ |
| IxAMAX   | ✔ | ✔ | ✔ | ✔ | ✔ |

| Level-2  | S | D | C | Z | H |
| ---------|---|---|---|---|---|
| xGEMV    | ✔ | ✔ | ✔ | ✔ | ✔ |
| xGBMV    | ✔ | ✔ | ✔ | ✔ | ✔ |
| xHEMV    | - | - | ✔ | ✔ | - |
| xHBMV    | - | - | ✔ | ✔ | - |
| xHPMV    | - | - | ✔ | ✔ | - |
| xSYMV    | ✔ | ✔ | - | - | ✔ |
| xSBMV    | ✔ | ✔ | - | - | ✔ |
| xSPMV    | ✔ | ✔ | - | - | ✔ |
| xTRMV    | ✔ | ✔ | ✔ | ✔ | ✔ |
| xTBMV    | ✔ | ✔ | ✔ | ✔ | ✔ |
| xTPMV    | ✔ | ✔ | ✔ | ✔ | ✔ |
| xGER     | ✔ | ✔ | - | - | ✔ |
| xGERU    | - | - | ✔ | ✔ | - |
| xGERC    | - | - | ✔ | ✔ | - |
| xHER     | - | - | ✔ | ✔ | - |
| xHPR     | - | - | ✔ | ✔ | - |
| xHER2    | - | - | ✔ | ✔ | - |
| xHPR2    | - | - | ✔ | ✔ | - |
| xSYR     | ✔ | ✔ | - | - | ✔ |
| xSPR     | ✔ | ✔ | - | - | ✔ |
| xSYR2    | ✔ | ✔ | - | - | ✔ |
| xSPR2    | ✔ | ✔ | - | - | ✔ |

| Level-3  | S | D | C | Z | H |
| ---------|---|---|---|---|---|
| xGEMM    | ✔ | ✔ | ✔ | ✔ | ✔ |
| xSYMM    | ✔ | ✔ | ✔ | ✔ | ✔ |
| xHEMM    | - | - | ✔ | ✔ | - |
| xSYRK    | ✔ | ✔ | ✔ | ✔ | ✔ |
| xHERK    | - | - | ✔ | ✔ | - |
| xSYR2K   | ✔ | ✔ | ✔ | ✔ | ✔ |
| xHER2K   | - | - | ✔ | ✔ | - |
| xTRMM    | ✔ | ✔ | ✔ | ✔ | ✔ |

In addition, some extra non-BLAS routines are also supported by CLBlast, classified as level-X. They are experimental and should be used with care:

| Level-X    | S | D | C | Z | H |
| -----------|---|---|---|---|---|
| xSUM       | ✔ | ✔ | ✔ | ✔ | ✔ |
| IxMAX      | ✔ | ✔ | ✔ | ✔ | ✔ |
| IxMIN      | ✔ | ✔ | ✔ | ✔ | ✔ |
| xOMATCOPY  | ✔ | ✔ | ✔ | ✔ | ✔ |

Some less commonly used BLAS routines are not yet supported yet by CLBlast. They are xROTG, xROTMG, xROT, xROTM, xTRSV, xTBSV, xTPSV, and xTRSM.


Half precision (fp16)
-------------

The half-precison fp16 format is a 16-bits floating-point data-type. Some OpenCL devices support the `cl_khr_fp16` extension, reducing storage and bandwidth requirements by a factor 2 compared to single-precision floating-point. In case the hardware also accelerates arithmetic on half-precision data-types, this can also greatly improve compute performance of e.g. level-3 routines such as GEMM. Devices which can benefit from this are among others Intel GPUs, ARM Mali GPUs, and NVIDIA's latest Pascal GPUs. Half-precision is in particular interest for the deep-learning community, in which convolutional neural networks can be processed much faster at a minor accuracy loss.

Since there is no half-precision data-type in C or C++, OpenCL provides the `cl_half` type for the host device. Unfortunately, internally this translates to a 16-bits integer, so computations on the host using this data-type should be avoided. For convenience, CLBlast provides the `clblast_half.h` header (C99 and C++ compatible), defining the `half` type as a short-hand to `cl_half` and the following basic functions:

* `half FloatToHalf(const float value)`: Converts a 32-bits floating-point value to a 16-bits floating-point value.
* `float HalfToFloat(const half value)`: Converts a 16-bits floating-point value to a 32-bits floating-point value.

The `samples/haxpy.c` example shows how to use these convencience functions when calling the half-precision BLAS routine HAXPY.


Contributing
-------------

Contributions are welcome in the form of tuning results for OpenCL devices previously untested. Furthermore, merge requests are welcome as long as they contain unit additions or modifications. Furthermore, they should follow the CLBlast coding style, which is based on the [Google C++ style guide](https://google-styleguide.googlecode.com/svn/trunk/cppguide.html) and the Effective C++ books by Scott Meyers.

The contributing authors (code, pull requests, testing) so far are:

* [Cedric Nugteren](http://www.cedricnugteren.nl) - main author
* [Anton Lokhmotov](https://github.com/psyhtest)
* [Dragan Djuric](https://github.com/blueberry)
* [Marco Hutter](https://github.com/gpus)
* [Hugh Perkins](https://github.com/hughperkins)
* [Gian-Carlo Pascutto](https://github.com/gcp)

Tuning and testing on a variety of OpenCL devices was made possible by:

* [TU/e ES research group](http://www.es.ele.tue.nl/)
* [ASCI DAS4 and DAS5](http://www.cs.vu.nl/das4/)
* [dividiti](http://www.dividiti.com)
* [SURFsara HPC center](http://www.surfsara.com)


Support us
-------------

This project started in March 2015 as an evenings and weekends free-time project next to a full-time job for Cedric Nugteren. If you are in the position to support the project by OpenCL-hardware donations or otherwise, please find contact information on the [website of the main author](http://www.cedricnugteren.nl).
