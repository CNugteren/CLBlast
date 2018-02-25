CLBlast: Performance measuring and benchmarking
================

This document describes how to measure the performance of CLBlast and how to compare it against other libraries. For other information about CLBlast, see the [main README](../README.md).


Compiling the performance tests ('clients')
-------------

To test the performance of CLBlast and to compare optionally against [clBLAS](http://github.com/clMathLibraries/clBLAS), cuBLAS (if testing on an NVIDIA GPU and `-DCUBLAS=ON` is set), or a CPU BLAS library (if installed), compile with the clients enabled by specifying `-DCLIENTS=ON`, for example as follows:

    cmake -DCLIENTS=ON ..

The performance tests come in the form of client executables named `clblast_client_xxxxx`, in which `xxxxx` is the name of a routine (e.g. `xgemm`). These clients take a bunch of configuration options and directly run CLBlast in a head-to-head performance test against optionally clBLAS and/or a CPU BLAS library. You can use the command-line options `-clblas 1`, `-cblas 1`, or `-cublas 1` to select a library to test against.


Benchmarking
-------------

On [the CLBlast website](https://cnugteren.github.io/clblast) you will find performance results for various devices. Performance is compared in this case against a tuned version of the clBLAS library and optionally also against cuBLAS. Such graphs can be generated automatically on your own device as well. First, compile CLBlast with the clients enabled (see above). Then, make sure your installation of the reference clBLAS is performance-tuned by running the `tune` executable (shipped with clBLAS). Finally, run the Python/Matplotlib graph-script found in `scripts/benchmark/benchmark.py`. For example, to generate the SGEMM PDF on device 1 of platform 0 from the `build` subdirectory:

    python ../scripts/benchmark/benchmark.py --platform 0 --device 1 --benchmark gemm

Note that the CLBlast library provides pre-tuned parameter-values for some devices only: if your device is not among these, then out-of-the-box performance might be poor. See the [tuning README](tuning.md) to find out how to tune for your device.

In case performance is still sub-optimal or something else is wrong, CLBlast can be build in verbose mode for (performance) debugging by specifying `-DVERBOSE=ON` to CMake.
