
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
__kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))
void CopyMatrixFast(const int ld,
                    __global const realC* restrict src,
                    __global realC* dest,
                    const real_arg arg_alpha) {
  const real alpha = GetRealArg(arg_alpha);
  #pragma unroll
  for (int _w_one = 0; _w_one < COPY_WPT; _w_one += 1) {
    const int id_one = get_global_id(0);
    const int id_two = (get_group_id(1)*COPY_WPT + _w_one) * COPY_DIMY + get_local_id(1);
    const int id = id_two*(ld/COPY_VW) + id_one;
    realC result;
    #if COPY_VW == 1
      Multiply(result, alpha, src[id]);
    #elif COPY_VW == 2
      Multiply(result.x, alpha, src[id].x);
      Multiply(result.y, alpha, src[id].y);
    #elif COPY_VW == 4
      Multiply(result.x, alpha, src[id].x);
      Multiply(result.y, alpha, src[id].y);
      Multiply(result.z, alpha, src[id].z);
      Multiply(result.w, alpha, src[id].w);
    #elif COPY_VW == 8
      Multiply(result.s0, alpha, src[id].s0);
      Multiply(result.s1, alpha, src[id].s1);
      Multiply(result.s2, alpha, src[id].s2);
      Multiply(result.s3, alpha, src[id].s3);
      Multiply(result.s4, alpha, src[id].s4);
      Multiply(result.s5, alpha, src[id].s5);
      Multiply(result.s6, alpha, src[id].s6);
      Multiply(result.s7, alpha, src[id].s7);
    #elif COPY_VW == 16
      Multiply(result.s0, alpha, src[id].s0);
      Multiply(result.s1, alpha, src[id].s1);
      Multiply(result.s2, alpha, src[id].s2);
      Multiply(result.s3, alpha, src[id].s3);
      Multiply(result.s4, alpha, src[id].s4);
      Multiply(result.s5, alpha, src[id].s5);
      Multiply(result.s6, alpha, src[id].s6);
      Multiply(result.s7, alpha, src[id].s7);
      Multiply(result.s8, alpha, src[id].s8);
      Multiply(result.s9, alpha, src[id].s9);
      Multiply(result.sA, alpha, src[id].sA);
      Multiply(result.sB, alpha, src[id].sB);
      Multiply(result.sC, alpha, src[id].sC);
      Multiply(result.sD, alpha, src[id].sD);
      Multiply(result.sE, alpha, src[id].sE);
      Multiply(result.sF, alpha, src[id].sF);
    #endif
    dest[id] = result;;
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
