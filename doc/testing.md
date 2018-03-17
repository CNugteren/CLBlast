CLBlast: Testing the library for correctness
================

This document describes how to test the library. For other information about CLBlast, see the [main README](../README.md).


Compiling the correctness tests
-------------

To make sure CLBlast is working correctly on your device (recommended), compile with the tests enabled by specifying `-DTESTS=ON`, for example as follows:

    cmake -DTESTS=ON ..

To build these tests, another BLAS library is needed to serve as a reference. This can be either:

* The OpenCL BLAS library [clBLAS](http://github.com/clMathLibraries/clBLAS) (maintained by AMD)
* A regular CPU Netlib BLAS library, e.g.:
  - OpenBLAS
  - BLIS
  - Accelerate

Afterwards, executables in the form of `clblast_test_xxxxx` are available, in which `xxxxx` is the name of a routine (e.g. `xgemm`). 


Running the tests
-------------

All tests can be run as individual executables or directly together in one go through the `make alltests` target or using CTest (`make test` or `ctest`). In the latter case the output is less verbose. Both cases allow you to set the default device and platform to non-zero by setting the `CLBLAST_DEVICE` and `CLBLAST_PLATFORM` environmental variables. Further options can be supplied through the `CLBLAST_ARGUMENTS` environmental variable (e.g. export CLBLAST_ARGUMENTS="-full_test -cblas 1 -clblas 0" on a UNIX system).

Note that CLBlast is tested for correctness against [clBLAS](http://github.com/clMathLibraries/clBLAS) and/or a regular CPU BLAS library. If both are installed on your system, setting the command-line option `-clblas 1` or `-cblas 1` will select the library to test against for the `clblast_test_xxxxx` executables. All tests have a `-verbose` option to enable additional diagnostic output. They also have a `-full_test` option to increase coverage further.
