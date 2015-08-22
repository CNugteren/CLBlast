
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xaxpy kernel. It contains one fast vectorized version in case of unit
// strides (incx=incy=1) and no offsets (offx=offy=0). Another version is more general, but doesn't
// support vector data-types.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Full version of the kernel with offsets and strided accesses
__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void Xaxpy(const int n, const real alpha,
                    const __global real* restrict xgm, const int x_offset, const int x_inc,
                    __global real* ygm, const int y_offset, const int y_inc) {

  // Loops over the work that needs to be done (allows for an arbitrary number of threads)
  #pragma unroll
  for (int id = get_global_id(0); id<n; id += get_global_size(0)) {
    MultiplyAdd(ygm[id*y_inc + y_offset], alpha, xgm[id*x_inc + x_offset]);
  }
}

// =================================================================================================

// Faster version of the kernel without offsets and strided accesses. Also assumes that 'n' is
// dividable by 'VW', 'WGS' and 'WPT'.
__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void XaxpyFast(const int n, const real alpha,
                         const __global realV* restrict xgm,
                         __global realV* ygm) {
  #pragma unroll
  for (int w=0; w<WPT; ++w) {
    const int id = w*get_global_size(0) + get_global_id(0);
    ygm[id] = MultiplyAddVector(ygm[id], alpha, xgm[id]);
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
