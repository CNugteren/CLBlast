CLBlast: Supported routines overview
================

This document describes which routines are supported in CLBlast. For other information about CLBlast, see the [main README](../README.md).

Full API documentation is available in a separate [API documentation file](api.md).


Supported types
-------------

The different data-types supported by the library are:

* __S:__ Single-precision 32-bit floating-point (`float`).
* __D:__ Double-precision 64-bit floating-point (`double`).
* __C:__ Complex single-precision 2x32-bit floating-point (`std::complex<float>`).
* __Z:__ Complex double-precision 2x64-bit floating-point (`std::complex<double>`).
* __H:__ Half-precision 16-bit floating-point (`cl_half`). See section 'Half precision' below for more information.


Supported routines
-------------

CLBlast supports almost all the Netlib BLAS routines plus a couple of extra non-BLAS routines. The supported BLAS routines are marked with '✔' in the following tables. Routines marked with '-' do not exist: they are not part of BLAS at all.

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
| xTRSV    | ✔ | ✔ | ✔ | ✔ |   |

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
| xTRSM    | ✔ | ✔ | ✔ | ✔ |   |

Furthermore, there are also batched versions of BLAS routines available, processing multiple smaller computations in one go for better performance:

| Batched             | S | D | C | Z | H |
| --------------------|---|---|---|---|---|
| xAXPYBATCHED        | ✔ | ✔ | ✔ | ✔ | ✔ |
| xGEMMBATCHED        | ✔ | ✔ | ✔ | ✔ | ✔ |
| xGEMMSTRIDEDBATCHED | ✔ | ✔ | ✔ | ✔ | ✔ |

In addition, some extra non-BLAS routines are also supported by CLBlast, classified as level-X. They are experimental and should be used with care:

| Level-X    | S | D | C | Z | H |
| -----------|---|---|---|---|---|
| xSUM       | ✔ | ✔ | ✔ | ✔ | ✔ | (Similar to xASUM, but not absolute)
| IxAMIN     | ✔ | ✔ | ✔ | ✔ | ✔ | (Similar to IxAMAX, but minimum instead of maximum)
| IxMAX      | ✔ | ✔ | ✔ | ✔ | ✔ | (Similar to IxAMAX, but not absolute)
| IxMIN      | ✔ | ✔ | ✔ | ✔ | ✔ | (Similar to IxAMAX, but not absolute and minimum instead of maximum)
| xHAD       | ✔ | ✔ | ✔ | ✔ | ✔ | (Hadamard product)
| xOMATCOPY  | ✔ | ✔ | ✔ | ✔ | ✔ | (Out-of-place copying/transposing/scaling of matrices)
| xIM2COL    | ✔ | ✔ | ✔ | ✔ | ✔ | (Image to column transform as used to express convolution as GEMM)
| xCOL2IM    | ✔ | ✔ | ✔ | ✔ | ✔ | (Column to image transform as used in machine learning)
| xCONVGEMM  | ✔ | ✔ | - | - | ✔ | (Experimental, implemented as either im2col followed by batched GEMM or as a single kernel)

Some less commonly used BLAS routines are not yet supported by CLBlast. They are xROTG, xROTMG, xROT, xROTM, xTBSV, and xTPSV.


Half precision (fp16)
-------------

The half-precision fp16 format is a 16-bits floating-point data-type. Some OpenCL devices support the `cl_khr_fp16` extension, reducing storage and bandwidth requirements by a factor 2 compared to single-precision floating-point. In case the hardware also accelerates arithmetic on half-precision data-types, this can also greatly improve compute performance of e.g. level-3 routines such as GEMM. Devices which can benefit from this are among others Intel GPUs, ARM Mali GPUs, and NVIDIA's latest Pascal GPUs. Half-precision is in particular interest for the deep-learning community, in which convolutional neural networks can be processed much faster at a minor accuracy loss.

Since there is no half-precision data-type in C or C++, OpenCL provides the `cl_half` type for the host device. Unfortunately, internally this translates to a 16-bits integer, so computations on the host using this data-type should be avoided. For convenience, CLBlast provides the `clblast_half.h` header (C99 and C++ compatible), defining the `half` type as a short-hand to `cl_half` and the following basic functions:

* `half FloatToHalf(const float value)`: Converts a 32-bits floating-point value to a 16-bits floating-point value.
* `float HalfToFloat(const half value)`: Converts a 16-bits floating-point value to a 32-bits floating-point value.

The [samples/haxpy.c](../samples/haxpy.c) example shows how to use these convenience functions when calling the half-precision BLAS routine HAXPY.
