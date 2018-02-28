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

If your device is not (yet) among this list or if you want to tune CLBlast for specific parameters (e.g. rectangular matrix sizes), you should run the included tuners.


Compiling and running the tuners
-------------

The included CLBlast tuners are compiled with the default CMake options. If they are not compiled, make sure you are specifing `-DTUNERS=ON`, for example as follows:

    cmake -DTUNERS=ON ..

Compiling with `-DTUNERS=ON` will generate a number of tuners, each named `clblast_tuner_xxxxx`, in which `xxxxx` corresponds to a `.opencl` kernel file as found in `src/kernels`. These kernels corresponds to routines (e.g. `xgemm`) or to common pre-processing or post-processing kernels (`copy` and `transpose`). Running such a tuner will test a number of parameter-value combinations on your device and report which one gave the best performance. Running `make alltuners` runs all tuners for all precisions in one go. You can set the default device and platform for `alltuners` by setting the `CLBLAST_DEVICE` and `CLBLAST_PLATFORM` environmental variables. Alternatively, you can also manually run each of the tuners for each of the precisions. Here's an example to tune the `axpy` kernels for 64-bit precision on device 0 of platform 0:

    ./clblast_tuner_xaxpy --precision 64 --device 0 --platform 0

The kernels `gemm` and `gemm_direct` have too many parameters to explore. Therefore, they will run in two stages: a first stage with a fixed limited number of parameter combinations, and a second stage with a random selection from a much larger search space. The random fraction is determined by the `fraction` argument on the command-line.

There are also several routine-level tuners. They tune inter-kernel parameters and should only be run after the kernels are tuned. An example is the GEMM routine tuner, which determines when to use the direct or the in-direct GEMM kernel.


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

After the kernels are tuned, you can run the `clblast_tuner_routine_xgemm` tuner to optimize the high-level GEMM routine, i.e. selecting which method to use: the direct kernel or the in-direct kernel.


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


Tuning OpenCL compiler options
-------------

For all of CLBlast's APIs, it is possible to optionally set an OS environmental variable `CLBLAST_BUILD_OPTIONS` to pass specific build options to the OpenCL compiler. Also make sure this is set in the same way when running the tuners.
