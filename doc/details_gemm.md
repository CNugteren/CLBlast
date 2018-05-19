CLBlast: Details on the GEMM routine and kernel
================

This document gives a bit more detail on how the GEMM routine is organised and implemented. For other information about CLBlast, see the [main README](../README.md).


GEMM: Two approaches
-------------

CLBlast implements two approaches to GEMM: direct and indirect:

* Direct GEMM: Computing GEMM using a single generic kernel which handles all cases (e.g. all kinds of matrix sizes).
* Indirect GEMM: Computing GEMM using multiple kernels: the main GEMM kernel and a few pre-processing and post-processing kernels. The main kernel makes several assumptions (e.g. sizes need to be multiples of 32), which the other kernels make sure are satisfied. The main kernel is often faster than the generic kernel of the direct approach, but the cost of pre-processing and post-processing kernels can sometimes be high for small sizes or particular devices.


GEMM: In-direct approach
-------------

Similar to the work by Matsumoto et al. ("Performance Tuning of Matrix Multiplication in OpenCL on Different GPUs and CPUs"), the main GEMM kernel makes many assumptions on the input arguments, which are handled by pre-processing and post-processing kernels. These assumptions are e.g. matrix sizes are a multiple of the work-group sizes, offsets are zero, and matrix B is transposed. This is a good solution for larger problem sizes since O(n^2) data movement is typically cheaper than O(n^3) computation, but the hidden constant starts to play a role for smaller n. Therefore, there is also a single-kernel direct version available for those cases, but it shares most of the design and parameters as discussed below.

The main kernel has 14 different parameters, of which some are illustrated in figure 1 in the [CLBlast paper](https://arxiv.org/pdf/1705.05249). The parameters define among others the work-group sizes in 2 dimensions (MWG, NWG), the 2D register tiling configuration (MWI, NWI), the vector widths of both input matrices (VWM, VWN), loop unroll factors (KWI), and whether or not and how to use the local memory.


GEMM: Direct approach
-------------

This is a single-kernel approach that shared many of the parameters for the in-direct kernel. One of the differences is that within the kernel there are checks for incomplete tiles in the m/n/k dimensions, influenced by the tuning parameters and the matrix sizes. These incomplete tiles will run a different part of the code, as they for example cannot benefit from vectorisation. Another difference is that there are dedicated kernels for each a/b transpose requirement: NN, NT, TN, TT for non-transposed and transposed.