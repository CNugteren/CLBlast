
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xgemv kernel for matrix-vector multiplication.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.
#ifndef WGS
  #define WGS 64     // The local work-group size
#endif
#ifndef WPT
  #define WPT 1      // The amount of work-per-thread
#endif
#ifndef VW
  #define VW 1       // Vector width of vectors X and Y
#endif

// =================================================================================================

// The gemv kernel
__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void Xgemv(const int m, const int n, const real alpha, const real beta,
                    const __global real* restrict agm,
                    const __global real* restrict xgm, const int x_offset, const int x_inc,
                    __global real* ygm, const int y_offset, const int y_inc) {

  // Loops over the work that needs to be done (allows for an arbitrary number of threads)
  #pragma unroll
  for (int id = get_global_id(0); id<m; id += get_global_size(0)) {

    // Loop over the elements of the matrix A
    real acc;
    SetToZero(acc);
    for (int k=0; k<n; ++k) {
      MultiplyAdd(acc, agm[id + m*k], xgm[k*x_inc + x_offset]);
    }
    AXPBY(ygm[id*y_inc + y_offset], alpha, acc, beta, ygm[id*y_inc + y_offset]);
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)";

// =================================================================================================
