CLBlast: Bindings / wrappers for other languages
================

The main APIs of CLBlast are C and C++ for OpenCL or CUDA. This document describes other APIs for other languages through bindings and wrappers. For other information about CLBlast, see the [main README](../README.md).


Plain C: Netlib BLAS API
-------------

CLBlast provides a Netlib CBLAS C API. This is however not recommended for performance, since at every call it will copy all buffers to and from the OpenCL device. Especially for level 1 and level 2 BLAS functions performance will be impacted severely. However, it can be useful if you don't want to touch OpenCL at all. Providing the `-DNETLIB=ON` flag to CMake at CLBlast compilation time will compile the Netlib API. Then, it can be used by including the corresponding header:

    #include <clblast_netlib_c.h>

The OpenCL device and platform can be set by setting the `CLBLAST_DEVICE` and `CLBLAST_PLATFORM` environmental variables. 


Python: PyCLBlast
-------------

PyCLBlast provides Python bindings for CLBlast. It is integrated in the main CLBlast project and can be installed through `pip`. Details can be found in the [PyCLBlast README](https://github.com/CNugteren/CLBlast/tree/master/src/pyclblast) or on [PyPi](https://pypi.python.org/pypi/pyclblast).


Java: JOCLBlast (3rd party)
-------------

JOCLBlast is a 3rd party project providing bindings for Java. It is built on top of JOCL. Details can be found on the [JOCLBlast Github project page](https://github.com/gpu/JOCLBlast).


Nim: nim-CLBlast (3rd party)
-------------

A 3rd party CLBlast wrapper for the nim language is available [here](https://github.com/numforge/nim-clblast).


Julia: CLBlast.jl (3rd party)
-------------

A 3rd party CLBlast wrapper for [Julia](https://julialang.org/) is available [here](https://github.com/JuliaGPU/CLBlast.jl).
