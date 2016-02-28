
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xher kernels for rank-1 matrix update.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Symmetric version of the rank-1 matrix update kernel (HER, HPR, SYR, SPR)
__attribute__((reqd_work_group_size(WGS1, WGS2, 1)))
__kernel void Xher(const int n, const real alpha,
                   const __global real* restrict xgm, const int x_offset, const int x_inc,
                   __global real* restrict agm, const int a_offset, const int a_ld,
                   const int is_upper, const int is_rowmajor) {

  // Register storage for X and XT
  real xvalues[WPT];
  real xtvalues[WPT];

  // Loads the X-vector
  #pragma unroll
  for (int w=0; w<WPT; ++w) {
    const int id2 = w*get_global_size(1) + get_global_id(1);
    xvalues[w] = LoadVector(id2, n, xgm, x_offset, x_inc, !is_rowmajor);
  }

  // Loads the X-transposed-vector
  #pragma unroll
  for (int w=0; w<WPT; ++w) {
    const int id1 = w*get_global_size(0) + get_global_id(0);
    xtvalues[w] = LoadVector(id1, n, xgm, x_offset, x_inc, is_rowmajor);
  }

  // Loops over the work per thread twice
  #pragma unroll
  for (int w1=0; w1<WPT; ++w1) {
    #pragma unroll
    for (int w2=0; w2<WPT; ++w2) {

      // Global thread IDs
      const int id1 = w1*get_global_size(0) + get_global_id(0);
      const int id2 = w2*get_global_size(1) + get_global_id(1);

      // Skip these threads if they do not contain threads contributing to the matrix-triangle
      if ((is_upper && (id1 > id2)) || (!is_upper && (id2 > id1))) {
        // Do nothing
      }

      // Loads A, performs the operation, and stores the result into A
      else {
        MatrixUpdate(id1, id2, n, n, agm, a_offset, a_ld, alpha, xvalues[w2], xtvalues[w1]);
      }
    }
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
