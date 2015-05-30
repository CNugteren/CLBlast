
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the common kernels shared among different BLAS routines. This file contains
// kernels to copy matrices.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.
#ifndef COPY_DIMX
  #define COPY_DIMX 8      // Local workgroup size in the first dimension (x)
#endif
#ifndef COPY_DIMY
  #define COPY_DIMY 8      // Local workgroup size in the second dimension (y)
#endif
#ifndef COPY_WPT
  #define COPY_WPT 1       // Work per thread in the first dimension (x)
#endif
#ifndef COPY_VW
  #define COPY_VW 1        // Vector width in the second dimension (y)
#endif

// =================================================================================================

// Data-widths
#if COPY_VW == 1
  typedef real realC;
#elif COPY_VW == 2
  typedef real2 realC;
#elif COPY_VW == 4
  typedef real4 realC;
#elif COPY_VW == 8
  typedef real8 realC;
#elif COPY_VW == 16
  typedef real16 realC;
#endif

// =================================================================================================

// Fast copy kernel. Requires 'ld' and the number of threads in dimension 0 to be a multiple of
// COPY_VW. Also requires both matrices to be of the same dimensions and without offset.
__attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))
__kernel void CopyMatrix(const int ld,
                         __global const realC* restrict src,
                         __global realC* dest) {
  #pragma unroll
  for (int w_one=0; w_one<COPY_WPT; ++w_one) {
    const int id_one = get_global_id(0);
    const int id_two = (get_group_id(1)*COPY_WPT + w_one) * COPY_DIMY + get_local_id(1);
    const int id = id_two*(ld/COPY_VW) + id_one;
    dest[id] = src[id];
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)";

// =================================================================================================
