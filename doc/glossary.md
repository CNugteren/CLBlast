CLBlast: Glossary
================

This document describes some commonly used terms in CLBlast documentation and code. For other information about CLBlast, see the [main README](../README.md).

* __BLAS__: The set of 'Basic Linear Algebra Subroutines'.
* __Netlib BLAS__: The official BLAS API definition, with __CBLAS__ providing the C headers. 
* __OpenCL__: The open compute language, a Khronos standard for heterogeneous and parallel computing, e.g. on GPUs.
* __kernel__: An OpenCL parallel program that runs on the target device.
* __clBLAS__: Another OpenCL BLAS library, maintained by AMD.
* __cuBLAS__: The main CUDA BLAS library, maintained by NVIDIA.
* __GEMM__: The 'GEneral Matrix Multiplication' routine.
* __Direct GEMM__: Computing GEMM using a single generic kernel which handles all cases (e.g. all kinds of matrix sizes).
* __Indirect GEMM__: Computing GEMM using multiple kernels: the main GEMM kernel and a few pre-processing and post-processing kernels. The main kernel makes several assumptions (e.g. sizes need to be multiples of 32), which the other kernels make sure are satisfied. The main kernel is often faster than the generic kernel of the direct approach, but the cost of pre-processing and post-processing kernels can sometimes be high for small sizes or particular devices.
