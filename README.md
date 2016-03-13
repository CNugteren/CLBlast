
CLBlast: The tuned OpenCL BLAS library
================

[![Build Status](https://travis-ci.org/CNugteren/CLBlast.svg?branch=master)](https://travis-ci.org/CNugteren/CLBlast)

CLBlast is a modern, lightweight, performant and tunable OpenCL BLAS library written in C++11. It is designed to leverage the full performance potential of a wide variety of OpenCL devices from different vendors, including desktop and laptop GPUs, embedded GPUs, and other accelerators. CLBlast implements BLAS routines: basic linear algebra subprograms operating on vectors and matrices.

__Note that the CLBlast library is actively being developed, and might not be mature enough for production environments__. This preview-version doesn't support the less commonly used routines yet: they will be added in due time. It also lacks extensive tuning on some common OpenCL platforms: __out-of-the-box performance on some devices might be poor__. See below for more details (and how to tune yourself).


Why CLBlast and not clBLAS or cuBLAS?
-------------

Use CLBlast instead of clBLAS:

* When you care about achieving maximum performance.
* When you want to be able to inspect the BLAS kernels or easily customize them to your needs.
* When you run on exotic OpenCL devices which you need to tune yourself.
* When you are still running on OpenCL 1.1 hardware.
* When you value an organized and modern C++ codebase.
* When you target Intel CPUs and GPUs or embedded devices

Use CLBlast instead of cuBLAS:

* When you want your code to run on devices other than NVIDIA CUDA-enabled GPUs.
* When you want to tune for a specific configuration (e.g. rectangular matrix-sizes)
* When you sleep better if you know that the library you use is open-source.

When not to use CLBlast:

* When you run on NVIDIA's CUDA-enabled GPUs only and can benefit from cuBLAS's assembly-level tuned kernels.
* When you need those BLAS routines that are not yet supported by CLBlast.


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

An example of an out-of-source build (starting from the root of the CLBlast folder):

    mkdir build
    cd build
    cmake ..
    make
    sudo make install

A custom installation folder can be specified when calling CMake:

    cmake -DCMAKE_INSTALL_PREFIX=/path/to/install/directory ..


Using the library
-------------

Like clBLAS and cuBLAS, CLBlast also requires OpenCL device buffers as arguments to its routines. This means you'll have full control over the OpenCL buffers and the host-device memory transfers. CLBlast's API is designed to resemble clBLAS's C API as much as possible, requiring little integration effort in case clBLAS was previously used. Using CLBlast starts by including the C++ header:

    #include <clblast.h>

Or alternatively the plain C version:

    #include <clblast_c.h>

Afterwards, any of CLBlast's routines can be called directly: there is no need to initialize the library. The available routines and the required arguments are described in the `clblast.h` include file. Additionally, a couple of stand-alone example programs are included in `samples/`.


Using the tuners (optional)
-------------

The CLBlast library will be tuned in the future for the most commonly used OpenCL devices. This pre-release of CLBlast is only tuned for a limited number of devices, in particular those with the following `CL_DEVICE_NAME` values:

* NVIDIA GPUs:
  - GeForce GTX 480
  - GeForce GTX 680
  - GeForce GTX 750 Ti
  - GeForce GTX 980
  - GeForce GTX Titan
  - GeForce GTX Titan X
  - Tesla K20m
  - Tesla K40m
* AMD GPUs:
  - Tahiti
  - R9 M370X
* Intel GPUs:
  - Iris
  - Iris Pro
* Intel CPUs:
  - Core i5-6200U
  - Core i7-3770K
  - Core i7-5930K
* Other devices:
  - ARM Mali-T628 GPU
  - Intel MIC

If your device is not (yet) among this list or if you want to tune CLBlast for specific parameters (e.g. rectangular matrix sizes), you should compile the library with the optional tuners:

    cmake -DTUNERS=ON ..

Note that CLBlast's tuners are based on the CLTune auto-tuning library, which has to be installed separately (version 1.7.0 or higher). CLTune is available from GitHub.

Compiling with `-DTUNERS=ON` will generate a number of tuners, each named `clblast_tuner_xxxxx`, in which `xxxxx` corresponds to a `.opencl` kernel file as found in `src/kernels`. These kernels corresponds to routines (e.g. `xgemm`) or to common pre-processing or post-processing kernels (`copy` and `transpose`). Running such a tuner will test a number of parameter-value combinations on your device and report which one gave the best performance. Running `make alltuners` runs all tuners for all precisions in one go. You can set the default device and platform for `alltuners` by setting the `DEFAULT_DEVICE` and `DEFAULT_PLATFORM` environmental variables before running CMake.

The tuners output a JSON-file with the results. The best results need to be added to `include/internal/database/xxxxx.h` in the appropriate section. However, this can be done automatically based on the JSON-data using a Python script in `scripts/database/database.py`. If you want the found parameters to be included in future releases of CLBlast, please attach the JSON files to the corresponding issue on GitHub or [email the main author](http://www.cedricnugteren.nl).

In summary, tuning the entire library for your device can be done as follows (starting from the root of the CLBlast folder):

    mkdir build
    cd build
    cmake -DTUNERS=ON ..
    make
    make alltuners
    python ../scripts/database/database.py . ..
    make


Compiling the tests (optional)
-------------

To make sure CLBlast is working correctly on your device (recommended), compile with the tests enabled:

    cmake -DTESTS=ON ..

Afterwards, executables in the form of `clblast_test_xxxxx` are available, in which `xxxxx` is the name of a routine (e.g. `xgemm`). Note that CLBlast is tested against [clBLAS](http://github.com/clMathLibraries/clBLAS) for correctness. The library clBLAS is therefore required to be installed on your system for the CLBlast tests.

With the `-DTESTS=ON` flag, additional performance tests are compiled. These come in the form of client executables named `clblast_client_xxxxx`, in which `xxxxx` is the name of a routine (e.g. `xgemm`). These clients take a bunch of configuration options and directly run both CLBlast and clBLAS in a head-to-head performance test.


Performance remarks
-------------

The CLBlast library provides pre-tuned parameter-values for a number of OpenCL devices. If your device is not among these, then out-of-the-box performance might be poor. Even if the device is included performance might be poor in some cases: __the preview version is not thoroughly tested for performance yet__. See above under `Using the tuners` to find out how to tune for your device.

The folder `doc/performance` contains some PDF files with performance results on tested devices. Performance is compared against a tuned version of the clBLAS library. The graphs of the level-3 routines (Xgemm, Xsymm, Xsyrk) show the strong points of CLBlast:

* The library reaches a high peak performance for large matrix sizes, in some cases a factor 2 more than clBLAS.
* The performance for non-power of 2 values (e.g. 1000) is roughly equal to power of 2 cases (e.g. 1024). This is not the case for clBLAS, which sometimes shows a drop of a factor 2.
* The performance is also constant for different layouts and transpose options. Again, this is not the case for clBLAS.

The graphs also show the current weak points of CLBlast: for small sizes the benefit is minimal or non-existent, and for some specific configurations clBLAS is still faster.

These graphs can be generated automatically on your own device. First, compile CLBlast with the tests enabled. Then, make sure your installation of the reference clBLAS is performance-tuned by running the `tune` executable. Finally, run one of the graph-scripts found in `test/performance/graphs` using R. For example, to generate the Xgemm PDF on device 1 of platform 0:

    Rscript path/to/test/performance/graphs/xgemm.r 0 1


Supported routines
-------------

CLBlast is in active development but already supports almost all the BLAS routines. The currently supported routines are marked with '✔' in the following tables. Empty boxes represent routines that still need to be implemented in a future release, whereas routines marked with '-' are not part of BLAS at all.

| Level-1  | S | D | C | Z | Notes   |
| ---------|---|---|---|---|---------|
| xROTG    |   |   | - | - |         |
| xROTMG   |   |   | - | - |         |
| xROT     |   |   | - | - |         |
| xROTM    |   |   | - | - |         |
| xSWAP    | ✔ | ✔ | ✔ | ✔ |         |
| xSCAL    | ✔ | ✔ | ✔ | ✔ | +CS +ZD |
| xCOPY    | ✔ | ✔ | ✔ | ✔ |         |
| xAXPY    | ✔ | ✔ | ✔ | ✔ |         |
| xDOT     | ✔ | ✔ | - | - |         |
| xDOTU    | - | - | ✔ | ✔ |         |
| xDOTC    | - | - | ✔ | ✔ |         |
| xNRM2    |   |   | - | - | +SC +DZ |
| xASUM    |   |   | - | - | +SC +DZ |
| IxAMAX   |   |   |   |   |         |

| Level-2  | S | D | C | Z | Notes   |
| ---------|---|---|---|---|---------|
| xGEMV    | ✔ | ✔ | ✔ | ✔ |         |
| xGBMV    | ✔ | ✔ | ✔ | ✔ |         |
| xHEMV    | - | - | ✔ | ✔ |         |
| xHBMV    | - | - | ✔ | ✔ |         |
| xHPMV    | - | - | ✔ | ✔ |         |
| xSYMV    | ✔ | ✔ | - | - |         |
| xSBMV    | ✔ | ✔ | - | - |         |
| xSPMV    | ✔ | ✔ | - | - |         |
| xTRMV    | ✔ | ✔ | ✔ | ✔ |         |
| xTBMV    | ✔ | ✔ | ✔ | ✔ |         |
| xTPMV    | ✔ | ✔ | ✔ | ✔ |         |
| xTRSV    |   |   |   |   |         |
| xTBSV    |   |   |   |   |         |
| xTPSV    |   |   |   |   |         |
| xGER     | ✔ | ✔ | - | - |         |
| xGERU    | - | - | ✔ | ✔ |         |
| xGERC    | - | - | ✔ | ✔ |         |
| xHER     | - | - | ✔ | ✔ |         |
| xHPR     | - | - | ✔ | ✔ |         |
| xHER2    | - | - | ✔ | ✔ |         |
| xHPR2    | - | - | ✔ | ✔ |         |
| xSYR     | ✔ | ✔ | - | - |         |
| xSPR     | ✔ | ✔ | - | - |         |
| xSYR2    | ✔ | ✔ | - | - |         |
| xSPR2    | ✔ | ✔ | - | - |         |

| Level-3  | S | D | C | Z | Notes   |
| ---------|---|---|---|---|---------|
| xGEMM    | ✔ | ✔ | ✔ | ✔ |         |
| xSYMM    | ✔ | ✔ | ✔ | ✔ |         |
| xHEMM    | - | - | ✔ | ✔ |         |
| xSYRK    | ✔ | ✔ | ✔ | ✔ |         |
| xHERK    | - | - | ✔ | ✔ |         |
| xSYR2K   | ✔ | ✔ | ✔ | ✔ |         |
| xHER2K   | - | - | ✔ | ✔ |         |
| xTRMM    | ✔ | ✔ | ✔ | ✔ |         |
| xTRSM    |   |   |   |   |         |


Contributing
-------------

Contributions are welcome in the form of tuning results for OpenCL devices previously untested. Furthermore, merge requests are welcome as long as they contain unit additions or modifications. Furthermore, they should follow the CLBlast coding style, which is based on the [Google C++ style guide](https://google-styleguide.googlecode.com/svn/trunk/cppguide.html) and the Effective C++ books by Scott Meyers.

The contributing authors so far are:

* [Cedric Nugteren](http://www.cedricnugteren.nl)

Tuning and testing on a variety of OpenCL devices was made possible by:

* [TU/e ES research group](http://www.es.ele.tue.nl/)
* [ASCI DAS4 and DAS5](http://www.cs.vu.nl/das4/)
* [Dividiti](http://www.dividiti.com)
* [SURFsara HPC center](http://www.surfsara.com)

Support us
-------------

This project started in March 2015 as an evenings and weekends free-time project next to a full-time job. If you are in the position to support the project by OpenCL-hardware donations or otherwise, please find contact information on the [website of the main author](http://www.cedricnugteren.nl).


To-do list before release of version 1.0
-------------

- Support all routines supported by clBLAS
- Allow the user control over events and synchronization
- Add half-precision routines (e.g. HGEMM)
- Enable correctness and performance testing against a CPU-based BLAS library
- Test in multi-threaded environments
