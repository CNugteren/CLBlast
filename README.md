
CLBlast: The tuned OpenCL BLAS library
================

| | Build status | Tests on Intel GPU | Tests on NVIDIA GPU | Tests on AMD GPU |
|-----|-----|-----|-----|-----|
| Windows | [![Build Status](https://ci.appveyor.com/api/projects/status/github/cnugteren/clblast?branch=master&svg=true)](https://ci.appveyor.com/project/CNugteren/clblast) | [![Build Status](http://ci.arrayfire.org/view/Other/job/other/job/CLBlast-Windows-Intel/badge/icon)](http://ci.arrayfire.org/view/Other/job/other/job/CLBlast-Windows-Intel/) | [![Build Status](http://ci.arrayfire.org/view/Other/job/other/job/CLBlast-Windows-NVIDIA/badge/icon)](http://ci.arrayfire.org/view/Other/job/other/job/CLBlast-Windows-NVIDIA/) | [![Build Status](http://ci.arrayfire.org/view/Other/job/other/job/CLBlast-Windows-AMD/badge/icon)](http://ci.arrayfire.org/view/Other/job/other/job/CLBlast-Windows-AMD/) |
| Linux | [![Build Status](https://travis-ci.org/CNugteren/CLBlast.svg?branch=master)](https://travis-ci.org/CNugteren/CLBlast/branches) | [![Build Status](http://ci.arrayfire.org/view/Other/job/other/job/CLBlast-Linux-Intel/badge/icon)](http://ci.arrayfire.org/view/Other/job/other/job/CLBlast-Linux-Intel/) | [![Build Status](http://ci.arrayfire.org/view/Other/job/other/job/CLBlast-Linux-NVIDIA/badge/icon)](http://ci.arrayfire.org/view/Other/job/other/job/CLBlast-Linux-NVIDIA/) | [![Build Status](http://ci.arrayfire.org/view/Other/job/other/job/CLBlast-Linux-AMD/badge/icon)](http://ci.arrayfire.org/view/Other/job/other/job/CLBlast-Linux-AMD/) |
| OS X | [![Build Status](https://travis-ci.org/CNugteren/CLBlast.svg?branch=master)](https://travis-ci.org/CNugteren/CLBlast/branches) | [![Build Status](http://ci.arrayfire.org/view/Other/job/other/job/CLBlast-OSX-Intel/badge/icon)](http://ci.arrayfire.org/view/Other/job/other/job/CLBlast-OSX-Intel/) | N/A | N/A |

(*Note*: Automated correctness tests currently not running, servers are offline)

CLBlast is a modern, lightweight, performant and tunable OpenCL BLAS library written in C++11. It is designed to leverage the full performance potential of a wide variety of OpenCL devices from different vendors, including desktop and laptop GPUs, embedded GPUs, and other accelerators. CLBlast implements BLAS routines: basic linear algebra subprograms operating on vectors and matrices. See [the CLBlast website](https://cnugteren.github.io/clblast) for performance reports on various devices as well as the latest CLBlast news.

The library is not tuned for all possible OpenCL devices: __if out-of-the-box performance is poor, please run the tuners first__. See below for a list of already tuned devices and instructions on how to tune yourself and contribute to future releases of the CLBlast library. See also the [CLBlast feature roadmap](ROADMAP.md) to get an indication of the future of CLBlast.


Why CLBlast and not clBLAS or cuBLAS?
-------------

Use CLBlast instead of clBLAS:

* When you care about achieving maximum performance.
* When you want to be able to inspect the BLAS kernels or easily customize them to your needs.
* When you run on exotic OpenCL devices for which you need to tune yourself.
* When you are still running on OpenCL 1.1 hardware.
* When you prefer a C++ API over a C API (C API also available in CLBlast).
* When you value an organized and modern C++ codebase.
* When you target Intel CPUs and GPUs or embedded devices.
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
  - MSVC (Visual Studio) 2013 or newer
* An OpenCL 1.1 or newer library, for example:
  - Apple OpenCL
  - NVIDIA CUDA SDK
  - AMD APP SDK
  - Intel OpenCL
  - Beignet
  - Mesa Clover
  - ARM Mali OpenCL

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

Building a static version of the library instead of shared one (.dylib/.so/.dll) can be done by disabling the `BUILD_SHARED_LIBS` option when calling CMake. For example:

    cmake -DBUILD_SHARED_LIBS=OFF ..


Using the library
-------------

Like clBLAS and cuBLAS, CLBlast also requires OpenCL device buffers as arguments to its routines. This means you'll have full control over the OpenCL buffers and the host-device memory transfers. CLBlast's API is designed to resemble clBLAS's C API as much as possible, requiring little integration effort in case clBLAS was previously used. Using CLBlast starts by including the C++ header:

    #include <clblast.h>

Or alternatively the plain C version:

    #include <clblast_c.h>

Afterwards, any of CLBlast's routines can be called directly: there is no need to initialize the library. The available routines and the required arguments are described in the above mentioned include files and the included [API documentation](doc/clblast.md). The API is kept as close as possible to the Netlib BLAS and the cuBLAS/clBLAS APIs.

To get started quickly, a couple of stand-alone example programs are included in the `samples` subfolder. They can optionally be compiled using the CMake infrastructure of CLBlast by providing the `-DSAMPLES=ON` flag, for example as follows:

    cmake -DSAMPLES=ON ..

For all of CLBlast's APIs, it is possible to optionally set an OS environmental variable `CLBLAST_BUILD_OPTIONS` to pass specific build options to the OpenCL compiler.


Using the library (Netlib API)
-------------

There is also a Netlib CBLAS C API available. This is however not recommended for full control over performance, since at every call it will copy all buffers to and from the OpenCL device. Especially for level 1 and level 2 BLAS functions performance will be impacted severely. However, it can be useful if you don't want to touch OpenCL at all. You can set the default device and platform by setting the `CLBLAST_DEVICE` and `CLBLAST_PLATFORM` environmental variables. This API can be used as follows after providing the `-DNETLIB=ON` flag to CMake:

    #include <clblast_netlib_c.h>


Using the library (CUDA API)
-------------

There is also a CUDA API of CLBlast available. Enabling this compiles the whole library for CUDA and thus replaces the OpenCL API. It is based upon the CUDA runtime and NVRTC APIs, requiring NVIDIA CUDA 7.5 or higher. The CUDA version of the library can be used as follows after providing the `-DCUDA=ON -DOPENCL=OFF` flags to CMake:

    #include <clblast_cuda.h>


Using the tuners (optional)
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
  - GeForce GTX 980
  - GeForce GTX 1070
  - GeForce GTX 1080
  - GeForce GTX 1080 Ti
  - GeForce GTX TITAN
  - GeForce GTX TITAN Black
  - GeForce GTX TITAN X
  - TITAN X (Pascal)
  - Tesla K20m
  - Tesla K40m
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
  - HD Graphics Haswell Ultrabook GT2 Mobile
  - HD Graphics IvyBridge M GT2
  - HD Graphics Skylake ULT GT2
  - Iris
  - Iris Pro
* Intel CPUs:
  - Core i5-4570
  - Core i5-6200U
  - Core i7-920
  - Core i7-2670QM
  - Core i7-3770K
  - Core i7-4790K
  - Core i7-5930K
  - Core i7-6770HQ
* Other devices:
  - ARM Mali-T628 GPU
  - ARM Mali-T760 GPU
  - Qualcomm Adreno 330 GPU
  - Intel MIC

If your device is not (yet) among this list or if you want to tune CLBlast for specific parameters (e.g. rectangular matrix sizes), you should compile the library with the optional tuners by specifing `-DTUNERS=ON`, for example as follows:

    cmake -DTUNERS=ON ..

Compiling with `-DTUNERS=ON` will generate a number of tuners, each named `clblast_tuner_xxxxx`, in which `xxxxx` corresponds to a `.opencl` kernel file as found in `src/kernels`. These kernels corresponds to routines (e.g. `xgemm`) or to common pre-processing or post-processing kernels (`copy` and `transpose`). Running such a tuner will test a number of parameter-value combinations on your device and report which one gave the best performance. Running `make alltuners` runs all tuners for all precisions in one go. You can set the default device and platform for `alltuners` by setting the `CLBLAST_DEVICE` and `CLBLAST_PLATFORM` environmental variables.

The tuners output a JSON-file with the results. The best results need to be added to `src/database/kernels/xxxxx.hpp` in the appropriate section. However, this can be done automatically based on the JSON-data using a Python (2.7 or 3.x) script in `scripts/database/database.py`. If you want the found parameters to be included in future releases of CLBlast, please attach the JSON files to the corresponding issue on GitHub or [email the main author](http://www.cedricnugteren.nl).

In summary, tuning the entire library for your device can be done as follows (starting from the root of the CLBlast folder):

    mkdir build
    cd build
    cmake -DTUNERS=ON ..
    make
    make alltuners
    python ../scripts/database/database.py . ..
    make

Alternatively, you can also supply your tuning parameters programmatically through the CLBlast API. This is especially useful if you tune for specific non-standard arguments (e.g. a rectangular or a very small matrix). To do so, you can call the `OverrideParameters` function which will set new parameters for a specific kernel. At the first next call of the target routine, CLBlast will compile a new binary and use it together with the new parameters from then on. Until `OverrideParameters` is called again of course. See the [API documentation](doc/clblast.md#overrideparameters-override-tuning-parameters-auxiliary-function) for more details.

After the kernels are tuned, you can run the `clblast_tuner_routine_xgemm` tuner to optimize the high-level GEMM routine, i.e. selecting which method to use: the direct kernel or the in-direct kernel.


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

All tests can be run directly together in one go through the `make alltests` target or using CTest (`make test` or `ctest`). In the latter case the output is less verbose. Both cases allow you to set the default device and platform to non-zero by setting the `CLBLAST_DEVICE` and `CLBLAST_PLATFORM` environmental variables. Further options can be supplied through the `CLBLAST_ARGUMENTS` environmental variable (e.g. export CLBLAST_ARGUMENTS="-full_test -cblas 1 -clblas 0" on a UNIX system).


Compiling the performance tests/clients (optional)
-------------

To test the performance of CLBlast and compare optionally against [clBLAS](http://github.com/clMathLibraries/clBLAS), cuBLAS (if testing on an NVIDIA GPU and `-DCUBLAS=ON` set), or a CPU BLAS library (see above for requirements), compile with the clients enabled by specifying `-DCLIENTS=ON`, for example as follows:

    cmake -DCLIENTS=ON ..

The performance tests come in the form of client executables named `clblast_client_xxxxx`, in which `xxxxx` is the name of a routine (e.g. `xgemm`). These clients take a bunch of configuration options and directly run CLBlast in a head-to-head performance test against optionally clBLAS and/or a CPU BLAS library. You can use the command-line options `-clblas 1` or `-cblas 1` to select a library to test against.

On [the CLBlast website](https://cnugteren.github.io/clblast) you will find performance results for various devices. Performance is compared in this case against a tuned version of the clBLAS library and optionally also against cuBLAS. Such graphs can be generated automatically on your own device as well. First, compile CLBlast with the clients enabled. Then, make sure your installation of the reference clBLAS is performance-tuned by running the `tune` executable (shipped with clBLAS). Finally, run the Python/Matplotlib graph-script found in `scripts/benchmark/benchmark.py`. For example, to generate the SGEMM PDF on device 1 of platform 0 from the `build` subdirectory:

    python ../scripts/benchmark/benchmark.py --platform 0 --device 1 --benchmark gemm

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
| xTRSV    | ✔ | ✔ | ✔ | ✔ |   | (experimental, un-optimized)

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
| xTRSM    | ✔ | ✔ | ✔ | ✔ |   | (experimental, un-optimized)

Furthermore, there are also batched versions of BLAS routines available, processing multiple smaller computations in one go for better performance:

| Batched             | S | D | C | Z | H |
| --------------------|---|---|---|---|---|
| xAXPYBATCHED        | ✔ | ✔ | ✔ | ✔ | ✔ |
| xGEMMBATCHED        | ✔ | ✔ | ✔ | ✔ | ✔ |
| xGEMMSTRIDEDBATCHED | ✔ | ✔ | ✔ | ✔ | ✔ |

In addition, some extra non-BLAS routines are also supported by CLBlast, classified as level-X. They are experimental and should be used with care:

| Level-X    | S | D | C | Z | H |
| -----------|---|---|---|---|---|
| xSUM       | ✔ | ✔ | ✔ | ✔ | ✔ |
| IxAMIN     | ✔ | ✔ | ✔ | ✔ | ✔ |
| IxMAX      | ✔ | ✔ | ✔ | ✔ | ✔ |
| IxMIN      | ✔ | ✔ | ✔ | ✔ | ✔ |
| xHAD       | ✔ | ✔ | ✔ | ✔ | ✔ | (Hadamard product)
| xOMATCOPY  | ✔ | ✔ | ✔ | ✔ | ✔ |
| xIM2COL    | ✔ | ✔ | ✔ | ✔ | ✔ |

Some less commonly used BLAS routines are not yet supported yet by CLBlast. They are xROTG, xROTMG, xROT, xROTM, xTBSV, and xTPSV.


Half precision (fp16)
-------------

The half-precision fp16 format is a 16-bits floating-point data-type. Some OpenCL devices support the `cl_khr_fp16` extension, reducing storage and bandwidth requirements by a factor 2 compared to single-precision floating-point. In case the hardware also accelerates arithmetic on half-precision data-types, this can also greatly improve compute performance of e.g. level-3 routines such as GEMM. Devices which can benefit from this are among others Intel GPUs, ARM Mali GPUs, and NVIDIA's latest Pascal GPUs. Half-precision is in particular interest for the deep-learning community, in which convolutional neural networks can be processed much faster at a minor accuracy loss.

Since there is no half-precision data-type in C or C++, OpenCL provides the `cl_half` type for the host device. Unfortunately, internally this translates to a 16-bits integer, so computations on the host using this data-type should be avoided. For convenience, CLBlast provides the `clblast_half.h` header (C99 and C++ compatible), defining the `half` type as a short-hand to `cl_half` and the following basic functions:

* `half FloatToHalf(const float value)`: Converts a 32-bits floating-point value to a 16-bits floating-point value.
* `float HalfToFloat(const half value)`: Converts a 16-bits floating-point value to a 32-bits floating-point value.

The `samples/haxpy.c` example shows how to use these convenience functions when calling the half-precision BLAS routine HAXPY.


Notes for Android
-------------

For deployment on Android, there are three options to consider.

First of all, you can use Google's recommended route of installing Android Studio with the NDK, and then use the JNI to interface to the CLBlast library. For this, we refer to the official Android Studio documentation and the online tutorials.

Alternatively, you can cross-compile the library and the test/client/tuner executables directly. To do so, first install the NDK, then find your vendor's OpenCL library (e.g. in `/system/vendor/lib`), get OpenCL headers from the Khronos registry, and invoke CMake as follows:

    cmake .. \
     -DCMAKE_SYSTEM_NAME=Android \
     -DCMAKE_SYSTEM_VERSION=19 \             # Set the appropriate Android API level
     -DCMAKE_ANDROID_ARCH_ABI=armeabi-v7a \  # Set the appropriate device architecture (e.g. armeabi-v7a or arm64-v8a)
     -DCMAKE_ANDROID_NDK=$ANDROID_NDK_PATH \ # Assumes $ANDROID_NDK_PATH points to your NDK installation
     -DCMAKE_ANDROID_STL_TYPE=gnustl_static \
     -DOPENCL_ROOT=/path/to/vendor/OpenCL/lib/folder/   # Should contain libOpenCL.so and CL/cl.h

For any potential issues, first check [cmath 'has not been declared' errors](https://stackoverflow.com/questions/45183525/compilation-error-with-ndk-using-cstatic/46433625). Also, if you are encountering errors such as `#error Bionic header ctype.h does not define either _U nor _CTYPE_U`, make sure CMake is not including system paths.

Finally, a third option is to use the [Collective Knowledge framework](https://github.com/ctuning/ck) in combination with the NDK, e.g. as follows:

    sudo pip install ck
    ck pull repo:ck-math
    ck install package:lib-clblast-master-universal --target_os=android21-arm64


Known issues
-------------

Known performance related issues:

* Severe performance issues with Beignet v1.3.0 due to missing support for local memory. Please downgrade to v1.2.1 or upgrade to v1.3.1 or newer.

* Performance issues on Qualcomm Adreno GPUs.

Other known issues:

* Routines returning an integer are currently not properly tested for half-precision FP16: IHAMAX/IHAMIN/IHMAX/IHMIN

* Half-precision FP16 tests might sometimes fail based on order multiplication, i.e. (a * b) * c != (c * b) * a

* The AMD APP SDK has a bug causing a conflict with libstdc++, resulting in a segfault when initialising static variables. This has been reported to occur with the CLBlast tuners.

* The AMD run-time compiler has a bug causing it to get stuck in an infinite loop. This is reported to happen occasionally when tuning the CLBlast GEMM routine.


Contributing
-------------

Contributions are welcome in the form of tuning results for OpenCL devices previously untested or pull requests. See [the contributing guidelines](CONTRIBUTING.md) for more details.

The main contributing authors (code, pull requests, testing) are:

* [Cedric Nugteren](http://cnugteren.github.io) - main author
* [Anton Lokhmotov](https://github.com/psyhtest)
* [Dragan Djuric](https://github.com/blueberry)
* [Marco Hutter](http://marco-hutter.de/)
* [Hugh Perkins](https://github.com/hughperkins)
* [Gian-Carlo Pascutto](https://github.com/gcp)
* [Ivan Shapovalov](https://github.com/intelfx)
* [Dimitri Van Assche](https://github.com/dvasschemacq)
* [Shehzan Mohammed](https://shehzan10.github.io)
* [Marco Cianfriglia](https://github.com/mcian)
* Everyone else listed as a [GitHub contributor](https://github.com/CNugteren/CLBlast/graphs/contributors)

Tuning and testing on a variety of OpenCL devices was made possible by:

* [TU/e ES research group](http://www.es.ele.tue.nl/)
* [ASCI DAS4 and DAS5](http://www.cs.vu.nl/das4/)
* [dividiti](http://www.dividiti.com)
* [SURFsara HPC center](http://www.surfsara.com)
* [ArrayFire](http://arrayfire.org)
* Everyone reporting [tuning results](https://github.com/CNugteren/CLBlast/issues/1)

Hardware/software for this project was contributed by:

* [ArrayFire](http://arrayfire.org) for settings up and supporting Jenkins CI correctness tests on 7 platforms
* [JetBrains](https://www.jetbrains.com/clion/) for supply a free CLion IDE license for CLBlast developers
* [Travis CI](https://travis-ci.org/CNugteren/CLBlast/branches) and [AppVeyor](https://ci.appveyor.com/project/CNugteren/clblast) for free automated build tests for open-source projects


More information
-------------

Further information on CLBlast is available through the following links:

* A 20-minute presentation of CLBlast was given at the GPU Technology Conference in May 2017. A recording is available on the [GTC on-demand website](http://on-demand.gputechconf.com/gtc/2017/video/s7280-nugteren-clblast.mp4) (poor audio quality however) and a full slide-set is also available [as PDF](http://on-demand.gputechconf.com/gtc/2017/presentation/s7280-cedric-nugteren-clblast.pdf).
* More in-depth information and experimental results are also available in a scientific paper titled [CLBlast: A Tuned OpenCL BLAS Library](https://arxiv.org/abs/1705.05249) (May 2017). For CLTune, the inspiration for the included auto-tuner, see also the [CLTune: A Generic Auto-Tuner for OpenCL Kernels](https://arxiv.org/abs/1703.06503) paper.


Support us
-------------

This project started in March 2015 as an evenings and weekends free-time project next to a full-time job for Cedric Nugteren. If you are in the position to support the project by OpenCL-hardware donations or otherwise, please find contact information on the [website of the main author](http://cnugteren.github.io).
