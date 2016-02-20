
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xger kernel (generic version) for rank-1 matrix update.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.

#ifndef WGS1
  #define WGS1 8    // The local work-group size in first dimension
#endif
#ifndef WGS2
  #define WGS2 8    // The local work-group size in second dimension
#endif
#ifndef WPT
  #define WPT 1     // The amount of work-per-thread in both dimensions
#endif

// =================================================================================================

// Row-major version of the kernel
__attribute__((reqd_work_group_size(WGS1, WGS2, 1)))
__kernel void Xger(const int max_one, const int max_two, const real alpha,
                   const __global real* restrict xgm, const int x_offset, const int x_inc,
                   const __global real* ygm, const int y_offset, const int y_inc,
                   __global real* restrict agm, const int a_offset, const int a_ld,
                   const int is_rowmajor) {

  // Register storage for X and Y
  real xvalues[WPT];
  real yvalues[WPT];

  // Row-major version
  if (is_rowmajor) {

    // Loads the X-vector
    #pragma unroll
    for (int w=0; w<WPT; ++w) {
      const int id2 = w*get_global_size(1) + get_global_id(1);
      if (id2 < max_two) {
        xvalues[w] = xgm[id2*x_inc + x_offset];
      }
    }

    // Loads the Y-vector
    #pragma unroll
    for (int w=0; w<WPT; ++w) {
      const int id1 = w*get_global_size(0) + get_global_id(0);
      if (id1 < max_one) {
        yvalues[w] = ygm[id1*y_inc + y_offset];
      }
    }

    // Loops over the work per thread twice
    #pragma unroll
    for (int w1=0; w1<WPT; ++w1) {
      #pragma unroll
      for (int w2=0; w2<WPT; ++w2) {

        // Global thread IDs
        const int id1 = w1*get_global_size(0) + get_global_id(0);
        const int id2 = w2*get_global_size(1) + get_global_id(1);

        if (id1 < max_one && id2 < max_two) {

          // Loads the current value of the A matrix
          const int a_index = id2*a_ld + id1 + a_offset;
          const real avalue = agm[a_index];

          // Computes result = alpha * x[i] * y[j] + a[i][j]
          real result;
          GER(result, alpha, xvalues[w2], yvalues[w1], avalue);
          
          // Stores the final result
          agm[a_index] = result;
        }
      }
    }
  }

  // Col-major version
  else {

    // Loads the X-vector
    #pragma unroll
    for (int w=0; w<WPT; ++w) {
      const int id1 = w*get_global_size(0) + get_global_id(0);
      if (id1 < max_one) {
        xvalues[w] = xgm[id1*x_inc + x_offset];
      }
    }

    // Loads the Y-vector
    #pragma unroll
    for (int w=0; w<WPT; ++w) {
      const int id2 = w*get_global_size(1) + get_global_id(1);
      if (id2 < max_two) {
        yvalues[w] = ygm[id2*y_inc + y_offset];
      }
    }

    // Loops over the work per thread twice
    #pragma unroll
    for (int w1=0; w1<WPT; ++w1) {
      #pragma unroll
      for (int w2=0; w2<WPT; ++w2) {

        // Global thread IDs
        const int id1 = w1*get_global_size(0) + get_global_id(0);
        const int id2 = w2*get_global_size(1) + get_global_id(1);

        if (id1 < max_one && id2 < max_two) {

          // Loads the current value of the A matrix
          const int a_index = id2*a_ld + id1 + a_offset;
          const real avalue = agm[a_index];

          // Computes result = alpha * x[i] * y[j] + a[i][j]
          real result;
          GER(result, alpha, xvalues[w1], yvalues[w2], avalue);
          
          // Stores the final result
          agm[a_index] = result;
        }
      }
    }
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
