
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xher2 kernels for rank-2 matrix update.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Symmetric version of the rank-2 matrix update kernel (HER2, HPR2, SYR2, SPR2)
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(WGS1, WGS2, 1)))
#endif
void Xher2(const int n,
           const real_arg arg_alpha,
           const __global real* restrict xgm, const int x_offset, const int x_inc,
           const __global real* restrict ygm, const int y_offset, const int y_inc,
           __global real* restrict agm, const int a_offset, const int a_ld,
           const int is_upper, const int is_rowmajor) {
  const real alpha = GetRealArg(arg_alpha);

  // Register storage for X and Y
  #pragma promote_to_registers
  real xvalues[WPT];
  #pragma promote_to_registers
  real yvalues[WPT];
  #pragma promote_to_registers
  real xtvalues[WPT];
  #pragma promote_to_registers
  real ytvalues[WPT];

  // Loads the X-vector
  #pragma unroll
  for (int _w = 0; _w < WPT; _w += 1) {
    const int id2 = _w*get_global_size(1) + get_global_id(1);
    xvalues[_w] = LoadVector(id2, n, xgm, x_offset, x_inc, !is_rowmajor);
  }

  // Loads the X-transposed-vector
  #pragma unroll
  for (int _w = 0; _w < WPT; _w += 1) {
    const int id1 = _w*get_global_size(0) + get_global_id(0);
    xtvalues[_w] = LoadVector(id1, n, xgm, x_offset, x_inc, is_rowmajor);
  }

  // Loads the Y-vector
  #pragma unroll
  for (int _w = 0; _w < WPT; _w += 1) {
    const int id1 = _w*get_global_size(0) + get_global_id(0);
    yvalues[_w] = LoadVector(id1, n, ygm, y_offset, y_inc, is_rowmajor);
  }

  // Loads the Y-transposed-vector
  #pragma unroll
  for (int _w = 0; _w < WPT; _w += 1) {
    const int id2 = _w*get_global_size(1) + get_global_id(1);
    ytvalues[_w] = LoadVector(id2, n, ygm, y_offset, y_inc, !is_rowmajor);
  }

  // Sets the proper value of alpha in case conjugation is needed
  real alpha1 = alpha;
  real alpha2 = alpha;
  #if defined(ROUTINE_HER2) || defined(ROUTINE_HPR2)
    if (is_rowmajor) {
      COMPLEX_CONJUGATE(alpha1);
    }
    else {
      COMPLEX_CONJUGATE(alpha2);
    }
  #endif

  // Loops over the work per thread twice
  #pragma unroll
  for (int _w1 = 0; _w1 < WPT; _w1 += 1) {
    #pragma unroll
    for (int _w2 = 0; _w2 < WPT; _w2 += 1) {

      // Global thread IDs
      const int id1 = _w1*get_global_size(0) + get_global_id(0);
      const int id2 = _w2*get_global_size(1) + get_global_id(1);

      // Skip these threads if they do not contain threads contributing to the matrix-triangle
      if ((is_upper && (id1 > id2)) || (!is_upper && (id2 > id1))) {
        // Do nothing
      }

      // Loads A, performs the operation, and stores the result into A
      else {
        MatrixUpdate2(id1, id2, n, n, agm, a_offset, a_ld,
                      alpha1, xvalues[_w2], yvalues[_w1],
                      alpha2, xtvalues[_w1], ytvalues[_w2], is_upper);
      }
    }
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
