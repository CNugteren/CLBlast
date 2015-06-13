
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
                    const int a_transposed,
                    const __global real* restrict agm, const int a_offset, const int a_ld,
                    const __global real* restrict xgm, const int x_offset, const int x_inc,
                    __global real* ygm, const int y_offset, const int y_inc) {

  // Local memory for the vector X
  __local real xlm[WGS];

  // Initializes the accumulation register
  real acc[WPT];
  #pragma unroll
  for (int w=0; w<WPT; ++w) {
    SetToZero(acc[w]);
  }

  // Divides the work in a main and tail section
  const int n_tail = n % WGS;
  const int n_floor = n - n_tail;

  // Loops over work-group sized portions of the work
  for (int kwg=0; kwg<n_floor; kwg+=WGS) {

    // Loads the vector X into local memory
    const int lid = get_local_id(0);
    xlm[lid] = xgm[(kwg + lid)*x_inc + x_offset];

    // Synchronizes all threads in a workgroup
    barrier(CLK_LOCAL_MEM_FENCE);

    // Loops over the work per thread
    #pragma unroll
    for (int w=0; w<WPT; ++w) {
      const int gid = w*get_global_size(0) + get_global_id(0);

      // Checks whether this thread is within bounds
      // Note: placed here because of the synchronisation barriers
      if (gid < m) {

        // Main multiply-add computation (regular)
        if (a_transposed == 0) {
          #pragma unroll
          for (int kl=0; kl<WGS; ++kl) {
            const int k = kwg + kl;
            MultiplyAdd(acc[w], agm[gid + a_ld*k + a_offset], xlm[kl]);
          }
        }

        // Main multiply-add computation (transposed)
        else {
          #pragma unroll
          for (int kl=0; kl<WGS; ++kl) {
            const int k = kwg + kl;
            MultiplyAdd(acc[w], agm[k + a_ld*gid + a_offset], xlm[kl]);
          }
        }
      }
    }

    // Synchronizes all threads in a workgroup
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Loops over the work per thread
  #pragma unroll
  for (int w=0; w<WPT; ++w) {
    const int gid = w*get_global_size(0) + get_global_id(0);

    // Checks whether this thread is within bounds
    if (gid < m) {

      // Multiply-add computation for the remaining tail (regular)
      if (a_transposed == 0) {
        #pragma unroll
        for (int k=n_floor; k<n; ++k) {
          MultiplyAdd(acc[w], agm[gid + a_ld*k + a_offset], xgm[k*x_inc + x_offset]);
        }
      }

      // Multiply-add computation for the remaining tail (transposed)
      else {
        #pragma unroll
        for (int k=n_floor; k<n; ++k) {
          MultiplyAdd(acc[w], agm[k + a_ld*gid + a_offset], xgm[k*x_inc + x_offset]);
        }
      }

      // Stores the final result
      AXPBY(ygm[gid*y_inc + y_offset], alpha, acc[w], beta, ygm[gid*y_inc + y_offset]);
    }
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)";

// =================================================================================================
