CLBlast: Details on the CONVGEMM routine
================

This document gives a bit more detail on how the CONVGEMM routine is organised and implemented. For other information about CLBlast, see the [main README](../README.md).


CONVGEMM: Two approaches
-------------

CLBlast implements two approaches to batched convolutions using GEMM: through im2col, or stand-alone:

* `ConvGemmMethod::kWithIm2Col`: running first a batched version of im2col to prepare the data into a temporary buffer, and then running a batched version of GEMM. The implementation is just as the regular im2col and GEMM kernels in CLBlast, but it is implemented as a separate kernel so all the non-needed features can be stripped out and some optimizations can be made. It uses the tuning parameters of the regular im2col and GEMM kernels.

* `ConvGemmMethod::kSingleKernel`: this is a single kernel approach: it loads the data in such a way that the im2col kernel is no longer needed, i.e. loading the data as the im2col transformation does it. That way it becomes a single kernel and there will be no need for an intermediate large buffer. It uses a separate set of tuning parameters, and can be tuned using the `clblast_tuner_xconvgemm` binary.


CONVGEMM: Selecting which approach to use
-------------

Since CONVGEMM is a relatively new and experimental feature, selection of the approach is hard-coded in [xconvgemm.hpp on line 32](../src/routines/levelx/xconvgemm.hpp:32), but can be changed there in a single place.

The main drawback of the `ConvGemmMethod::kWithIm2Col` approach is its extra memory usage, but depending on the device and setting, it might be faster compared to the `ConvGemmMethod::kSingleKernel` approach. The latter has as extra advantage that it has its own tuning parameters, so it can be fine-tuned for your specific use-case a bit better than the 2-kernel approach with im2col.
