
CLBlast: The tuned OpenCL BLAS library
================

CLBlast is a modern, lightweight, performant and tunable OpenCL BLAS library written in C++11. It is designed to leverage the full performance potential of a wide variety of OpenCL devices from different vendors, including desktop and laptop GPUs, embedded GPUs, and other accelerators. CLBlast implements BLAS routines: basic linear algebra subprograms operating on vectors and matrices.

__Note that the CLBlast library is actively being developed, and is not mature enough for production environments__. This preview-version supports only a minimal amount of routines (including `gemm` and `gemv`): others will be added in due time. It also lacks extensive tuning and testing on some common OpenCL platforms: __out-of-the-box performance on some devices might be poor__. See below for more details.


Why CLBlast and not clBLAS or cuBLAS?
-------------

Use CLBlast instead of clBLAS:

* When you care about achieving maximum performance.
* When you want to be able to inspect the BLAS kernels or easily customize them to your needs.
* When you run on exotic OpenCL devices which you need to tune yourself.

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
* An OpenCL 1.1 or newer library. CLBlast has been tested on x86-64 Linux and OS X systems with:
  - Apple OpenCL
  - NVIDIA CUDA SDK (5.5, 6.5, 7.0)
  - AMD APP SDK (2.9)

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

Like clBLAS and cuBLAS, CLBlast also requires OpenCL device buffers as arguments to its routines. This means you'll have full control over the OpenCL buffers and the host-device memory transfers. CLBlast's API is designed to resemble clBLAS's C API as much as possible, requiring little integration effort in case clBLAS was previously used. Using CLBlast starts by including the header:

    #include <clblast.h>

Afterwards, any of CLBlast's routines can be called directly: there is no need to initialize the library. The available routines and the required arguments are described in the `clblast.h` include file. Additionally, a couple of stand-alone sample program are included in `samples/`.


Using the tuners (optional)
-------------

The CLBlast library will be tuned in the future for the most commonly used OpenCL devices. This pre-release of CLBlast is only tuned for a limited number of devices, in particular those with the following `CL_DEVICE_NAME` values:

* NVIDIA GPUs:
  - GeForce GTX480
  - Tesla K20m
  - Tesla K40m
* AMD GPUs:
  - Tahiti
* Intel GPUs:
  - Iris

If your device is not (yet) among this list or if you want to tune CLBlast for specific parameters (e.g. rectangular matrix sizes), you should compile the library with the optional tuners:

    cmake -DTUNERS=ON ..

Note that CLBlast's tuners are based on the CLTune auto-tuning library, which has to be installed separately. CLTune is available from GitHub.

Compiling with `-DTUNERS=ON` will generate a number of tuners, each named `tuner_xxxxx`, in which `xxxxx` corresponds to a `.opencl` kernel file as found in `src/kernels`. These kernels corresponds to routines (e.g. `xgemm`) or to common pre-processing or post-processing kernels (`copy` and `transpose`). Running such a tuner will test a number of parameter-value combinations on your device and report which one gave the best performance.

The tuner will output a C++ database compatible line with the results, which can be added to `include/internal/database/xxxxx.h` in the appropriate section. Or, if tuning parameters already exist for your device but you believe they can be improved, this is also the place where they can be modified. If you want the found parameters to be included in future releases of CLBlast, please post the results in the corresponding issue on GitHub or [email the main author](http://www.cedricnugteren.nl).


Compiling the tests (optional)
-------------

To make sure CLBlast is working correctly on your device (recommended), compile with the tests enabled:

    cmake -DTESTS=ON ..

Afterwards, executables in the form of `test_xxxxx` are available, in which `xxxxx` is the name of a routine (e.g. `xgemm`). Note that CLBlast is tested against [clBLAS](http://github.com/clMathLibraries/clBLAS) for correctness. However, it is not required to install clBLAS separately on your system: it is included as part of the CLBlast source code in `external/clBLAS`.

With the `-DTESTS=ON` flag, additional performance tests are compiled. These come in the form of client executables named `client_xxxxx`, in which `xxxxx` is the name of a routine (e.g. `xgemm`). These clients take a bunch of configuration options and directly run both CLBlast and clBLAS in a head-to-head performance test.


Performance remarks
-------------

The CLBlast library provides pre-tuned parameter-values for a number of OpenCL devices. If your device is not among these, then out-of-the-box performance might be poor. Even if the device is included performance might be poor in some cases: __the preview version is not thoroughly tested for performance yet__. See above under `Using the tuners` to find out how to tune for your device.

The folder `doc/performance` contains some PDF files with performance results on tested devices. Performance is compared against a tuned version of the clBLAS library. The graphs of the level-3 routines (Xgemm and Xsymm) show the strong points of CLBlast:

* The library reaches a high peak performance for large matrix sizes, in some cases a factor 2 more than clBLAS.
* The performance for non-power of 2 values (e.g. 1000) is roughly equal to power of 2 cases (e.g. 1024). This is not the case for clBLAS, which sometimes shows a drop of a factor 2.
* The performance is also constant for different layouts and transpose options. Again, this is not the case for clBLAS.

The graphs also show the current weak point of CLBlast: its performance for smaller matrix sizes is not too good. Furthermore, although the GEMM kernels perform well on AMD GPUs, the supporting copy and transpose kernel do not.

These graphs can be generated automatically on your own device. First, compile CLBlast with the tests enabled. Then, make sure your installation of the reference clBLAS is performance-tuned by running the `tune` executable. Finally, run one of the graph-scripts found in `test/performance/graphs` using R. For example, to generate the Xgemm PDF on device 1 of platform 0:

    Rscript path/to/test/performance/graphs/xgemm.r 0 1

Supported routines
-------------

CLBlast is in active development and currently does not support the full set of BLAS routines. The currently supported routines are marked with '✔' in the following tables:

| Level-1  | S | D | C | Z | Notes   |
| ---------|---|---|---|---|---------|
| xROTG    |   |   | - | - |         |
| xROTMG   |   |   | - | - |         |
| xROT     |   |   | - | - |         |
| xROTM    |   |   | - | - |         |
| xSWAP    |   |   |   |   |         |
| xSCAL    |   |   |   |   | +CS +ZD |
| xCOPY    |   |   |   |   |         |
| xAXPY    | ✔ | ✔ | ✔ | ✔ |         |
| xDOT     |   |   | - | - | +DS     |
| xDOTU    | - | - |   |   |         |
| xDOTC    | - | - |   |   |         |
| xxxDOT   | - | - | - | - | +SDS    |
| xNRM2    |   |   | - | - | +SC +DZ |
| xASUM    |   |   | - | - | +SC +DZ |
| IxAMAX   |   |   |   |   |         |


| Level-2  | S | D | C | Z | Notes   |
| ---------|---|---|---|---|---------|
| xGEMV    | ✔ | ✔ | ✔ | ✔ |         |
| xGBMV    |   |   |   |   |         |
| xHEMV    | - | - |   |   |         |
| xHBMV    | - | - |   |   |         |
| xHPMV    | - | - |   |   |         |
| xSYMV    |   |   | - | - |         |
| xSBMV    |   |   | - | - |         |
| xSPMV    |   |   | - | - |         |
| xTRMV    |   |   |   |   |         |
| xTBMV    |   |   |   |   |         |
| xTPMV    |   |   |   |   |         |
| xTRSV    |   |   |   |   |         |
| xTBSV    |   |   |   |   |         |
| xTPSV    |   |   |   |   |         |
| xGER     |   |   | - | - |         |
| xGERU    | - | - |   |   |         |
| xGERC    | - | - |   |   |         |
| xHER     | - | - |   |   |         |
| xHPR     | - | - |   |   |         |
| xHER2    | - | - |   |   |         |
| xHPR2    | - | - |   |   |         |
| xSYR     |   |   | - | - |         |
| xSPR     |   |   | - | - |         |
| xSYR2    |   |   | - | - |         |
| xSPR2    |   |   | - | - |         |

| Level-3  | S | D | C | Z | Notes   |
| ---------|---|---|---|---|---------|
| xGEMM    | ✔ | ✔ | ✔ | ✔ |         |
| xSYMM    | ✔ | ✔ | ✔ | ✔ |         |
| xHEMM    | - | - |   |   |         |
| xSYRK    | ✔ | ✔ | ✔ | ✔ |         |
| xHERK    | - | - |   |   |         |
| xSYR2K   | ✔ | ✔ | ✔ | ✔ |         |
| xHER2K   | - | - |   |   |         |
| xTRMM    |   |   |   |   |         |
| xTRSM    |   |   |   |   |         |


Contributing
-------------

Contributions are welcome in the form of tuning results for OpenCL devices previously untested. Furthermore, merge requests are welcome as long as they contain unit additions or modifications. Furthermore, they should follow the CLBlast coding style, which is based on the [Google C++ style guide](https://google-styleguide.googlecode.com/svn/trunk/cppguide.html) and the Effective C++ books by Scott Meyers.

The contributing authors so far are:

* [Cedric Nugteren](http://www.cedricnugteren.nl)


Support us
-------------

This project started in March 2015 as an evenings and weekends free-time project next to a full-time job. If you are in the position to support the project by OpenCL-hardware donations or otherwise, please find contact information on the [website of the main author](http://www.cedricnugteren.nl).


To-do list before release of version 1.0
-------------

- Increase the functionality:
  * Support all routines supported by clBLAS
  * Allow the user control over events and synchronization
  * Add an interface with OpenCL C++ data-types
  * Add an old-style C compatible interface
  * Add half-precision routines (e.g. HGEMM)
- Improve host performance:
  * Allow initialization to pre-compile kernels and store to disk
- Improve device performance:
  * Enable 'mad()' for AMD devices
  * Improve the performance of the copy and transpose kernels
  * Tune for a wider range of devices
  * Allow users to define custom tuned parameters
- Improve the tuning
  * Make the tuners upload their data to a central server
- Improve the performance comparisons:
  * Enable comparison against optionally: ViennaCL, cuBLAS, MAGMA OpenCL
- Further reduce the likelihood of crashes:
  * Add checks for proper command-line arguments in the tuner, tester and client
  * Add checks for valid database parameters
  * Test in multi-threaded environments
