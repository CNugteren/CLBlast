
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

// The multiply-add function for the main part (divisable by WGS)
inline void MatrixVectorMain(const __global real* restrict agm, __local real* xlm, real acc[WPT],
                             const int gid, const int w, const int kwg,
                             const int a_ld, const int a_offset, const int a_transposed) {
  if (a_transposed == 0) { // Not transposed
    #pragma unroll
    for (int kl=0; kl<WGS; ++kl) {
      const int k = kwg + kl;
      MultiplyAdd(acc[w], agm[gid + a_ld*k + a_offset], xlm[kl]);
    }
  }
  else { // Transposed
    #pragma unroll
    for (int kl=0; kl<WGS; ++kl) {
      const int k = kwg + kl;
      MultiplyAdd(acc[w], agm[k + a_ld*gid + a_offset], xlm[kl]);
    }
  }
}

// The multiply-add function for the remainder part (not divisable by WGS)
inline void MatrixVectorRemainder(const __global real* restrict agm,
                                  const __global real* restrict xgm, real acc[WPT],
                                  const int gid, const int w, const int n_floor, const int n,
                                  const int a_ld, const int a_offset, const int a_transposed,
                                  const int x_inc, const int x_offset) {
  if (a_transposed == 0) { // Not transposed
    #pragma unroll
    for (int k=n_floor; k<n; ++k) {
      MultiplyAdd(acc[w], agm[gid + a_ld*k + a_offset], xgm[k*x_inc + x_offset]);
    }
  }
  else { // Transposed
    #pragma unroll
    for (int k=n_floor; k<n; ++k) {
      MultiplyAdd(acc[w], agm[k + a_ld*gid + a_offset], xgm[k*x_inc + x_offset]);
    }
  }
}

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

    // Loops over the work per thread, and checks whether in bounds
    #pragma unroll
    for (int w=0; w<WPT; ++w) {
      const int gid = w*get_global_size(0) + get_global_id(0);
      if (gid < m) {
        MatrixVectorMain(agm, xlm, acc, gid, w, kwg, a_ld, a_offset, a_transposed);
      }
    }

    // Synchronizes all threads in a workgroup
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Loops over the work per thread, and checks whether in bounds
  #pragma unroll
  for (int w=0; w<WPT; ++w) {
    const int gid = w*get_global_size(0) + get_global_id(0);
    if (gid < m) {
      MatrixVectorRemainder(agm, xgm, acc, gid, w, n_floor, n,
                            a_ld, a_offset, a_transposed, x_inc, x_offset);

      // Stores the final result
      AXPBY(ygm[gid*y_inc + y_offset], alpha, acc[w], beta, ygm[gid*y_inc + y_offset]);
    }
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)";

// =================================================================================================
