
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains common functions for matrix update kernels (Xger, Xher).
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

// Returns an element from a vector
inline real LoadVector(const int id, const int max,
                       __global real* restrict gm, const int offset, const int inc,
                       const int do_conjugate) {
  if (id < max) {
    real result = gm[id*inc + offset];
    if (do_conjugate) {
      #if defined(ROUTINE_GERC)
        COMPLEX_CONJUGATE(result);
      #endif
      #if defined(ROUTINE_HER)
        COMPLEX_CONJUGATE(result);
      #endif
    }
    return result;
  }
  else {
    real default_result;
    SetToZero(default_result);
    return default_result;
  }
}

// Performs the rank-1 matrix update
inline void MatrixUpdate(const int id1, const int id2, const int max1, const int max2,
                         __global real* restrict agm, const int a_offset, const int a_ld,
                         const real alpha, const real xvalue, const real yvalue) {

  // Bounds of a regular matrix
  if (id1 < max1 && id2 < max2) {

    #if defined(ROUTINE_SPR) || defined(ROUTINE_HPR)
      const int a_index = (id1 <= id2) ? ((id2+1)*id2)/2 + id1 + a_offset : ((id1+1)*id1)/2 + id2 + a_offset;
    #else
      const int a_index = id2*a_ld + id1 + a_offset;
    #endif

    // Loads the current value of the A matrix
    const real avalue = agm[a_index];

    // Computes result = alpha * x[i] * y[j] + a[i][j]
    real result;
    GER(result, alpha, xvalue, yvalue, avalue);

    // For hermetian matrices
    #if defined(ROUTINE_HER) || defined(ROUTINE_HPR)
      if (id1 == id2) { result.y = ZERO; }
    #endif
    
    // Stores the final result
    agm[a_index] = result;
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
