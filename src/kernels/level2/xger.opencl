
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xger kernels for rank-1 matrix update.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Regular version of the rank-1 matrix update kernel (GER, GERU, GERC)
__kernel __attribute__((reqd_work_group_size(WGS1, WGS2, 1)))
void Xger(const int max1, const int max2,
          const real_arg arg_alpha,
          const __global real* restrict xgm, const int x_offset, const int x_inc,
          const __global real* ygm, const int y_offset, const int y_inc,
          __global real* restrict agm, const int a_offset, const int a_ld,
          const int is_rowmajor) {
  const real alpha = GetRealArg(arg_alpha);

  // Register storage for X and Y
  #pragma promote_to_registers
  real xvalues[WPT];
  #pragma promote_to_registers
  real yvalues[WPT];

  // Row-major version
  if (is_rowmajor) {

    // Loads the X-vector
    #pragma unroll
    for (int _w = 0; _w < WPT; _w += 1) {
      const int id2 = _w*get_global_size(1) + get_global_id(1);
      xvalues[_w] = LoadVector(id2, max2, xgm, x_offset, x_inc, false);
    }

    // Loads the Y-vector
    #pragma unroll
    for (int _w = 0; _w < WPT; _w += 1) {
      const int id1 = _w*get_global_size(0) + get_global_id(0);
      yvalues[_w] = LoadVector(id1, max1, ygm, y_offset, y_inc, true);
    }

    // Loops over the work per thread twice
    #pragma unroll
    for (int _w1 = 0; _w1 < WPT; _w1 += 1) {
      #pragma unroll
      for (int _w2 = 0; _w2 < WPT; _w2 += 1) {

        // Global thread IDs
        const int id1 = _w1*get_global_size(0) + get_global_id(0);
        const int id2 = _w2*get_global_size(1) + get_global_id(1);

        // Loads A, performs the operation, and stores the result into A
        MatrixUpdate(id1, id2, max1, max2, agm, a_offset, a_ld,
                     alpha, xvalues[_w2], yvalues[_w1], false);
      }
    }
  }

  // Col-major version
  else {

    // Loads the X-vector
    #pragma unroll
    for (int _w = 0; _w < WPT; _w += 1) {
      const int id1 = _w*get_global_size(0) + get_global_id(0);
      xvalues[_w] = LoadVector(id1, max1, xgm, x_offset, x_inc, false);
    }

    // Loads the Y-vector
    #pragma unroll
    for (int _w = 0; _w < WPT; _w += 1) {
      const int id2 = _w*get_global_size(1) + get_global_id(1);
      yvalues[_w] = LoadVector(id2, max2, ygm, y_offset, y_inc, true);
    }

    // Loops over the work per thread twice
    #pragma unroll
    for (int _w1 = 0; _w1 < WPT; _w1 += 1) {
      #pragma unroll
      for (int _w2 = 0; _w2 < WPT; _w2 += 1) {

        // Global thread IDs
        const int id1 = _w1*get_global_size(0) + get_global_id(0);
        const int id2 = _w2*get_global_size(1) + get_global_id(1);

        // Loads A, performs the operation, and stores the result into A
        MatrixUpdate(id1, id2, max1, max2, agm, a_offset, a_ld,
                     alpha, xvalues[_w1], yvalues[_w2], false);
      }
    }
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
