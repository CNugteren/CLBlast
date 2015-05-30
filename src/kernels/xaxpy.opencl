
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xaxpy kernel. It contains one fast vectorized version in case of unit
// strides (incx=incy=1) and no offsets (offx=offy=0). Another version is more general, but doesn't
// support vector data-types.
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

// Data-widths
#if VW == 1
  typedef real realV;
#elif VW == 2
  typedef real2 realV;
#elif VW == 4
  typedef real4 realV;
#elif VW == 8
  typedef real8 realV;
#elif VW == 16
  typedef real16 realV;
#endif

// =================================================================================================

// The vectorized multiply-add function
inline realV MultiplyAddVector(realV cvec, const real aval, const realV bvec) {
  #if VW == 1
    MultiplyAdd(cvec, aval, bvec);
  #elif VW == 2
    MultiplyAdd(cvec.x, aval, bvec.x);
    MultiplyAdd(cvec.y, aval, bvec.y);
  #elif VW == 4
    MultiplyAdd(cvec.x, aval, bvec.x);
    MultiplyAdd(cvec.y, aval, bvec.y);
    MultiplyAdd(cvec.z, aval, bvec.z);
    MultiplyAdd(cvec.w, aval, bvec.w);
  #elif VW == 8
    MultiplyAdd(cvec.s0, aval, bvec.s0);
    MultiplyAdd(cvec.s1, aval, bvec.s1);
    MultiplyAdd(cvec.s2, aval, bvec.s2);
    MultiplyAdd(cvec.s3, aval, bvec.s3);
    MultiplyAdd(cvec.s4, aval, bvec.s4);
    MultiplyAdd(cvec.s5, aval, bvec.s5);
    MultiplyAdd(cvec.s6, aval, bvec.s6);
    MultiplyAdd(cvec.s7, aval, bvec.s7);
  #elif VW == 16
    MultiplyAdd(cvec.s0, aval, bvec.s0);
    MultiplyAdd(cvec.s1, aval, bvec.s1);
    MultiplyAdd(cvec.s2, aval, bvec.s2);
    MultiplyAdd(cvec.s3, aval, bvec.s3);
    MultiplyAdd(cvec.s4, aval, bvec.s4);
    MultiplyAdd(cvec.s5, aval, bvec.s5);
    MultiplyAdd(cvec.s6, aval, bvec.s6);
    MultiplyAdd(cvec.s7, aval, bvec.s7);
    MultiplyAdd(cvec.s8, aval, bvec.s8);
    MultiplyAdd(cvec.s9, aval, bvec.s9);
    MultiplyAdd(cvec.sA, aval, bvec.sA);
    MultiplyAdd(cvec.sB, aval, bvec.sB);
    MultiplyAdd(cvec.sC, aval, bvec.sC);
    MultiplyAdd(cvec.sD, aval, bvec.sD);
    MultiplyAdd(cvec.sE, aval, bvec.sE);
    MultiplyAdd(cvec.sF, aval, bvec.sF);
  #endif
  return cvec;
}

// =================================================================================================

// Full version of the kernel with offsets and strided accesses
__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void Xaxpy(const int n, const real alpha,
                    const __global real* restrict xgm, const int x_offset, const int x_inc,
                    __global real* ygm, const int y_offset, const int y_inc) {

  // Loops over the work that needs to be done (allows for an arbitrary number of threads)
  #pragma unroll
  for (int id = get_global_id(0); id<n; id += get_global_size(0)) {
    MultiplyAdd(ygm[id*y_inc + y_offset], alpha, xgm[id*x_inc + x_offset]);
  }
}

// =================================================================================================

// Faster version of the kernel without offsets and strided accesses. Also assumes that 'n' is
// dividable by 'VW', 'WGS' and 'WPT'.
__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void XaxpyFast(const int n, const real alpha,
                         const __global realV* restrict xgm,
                         __global realV* ygm) {
  #pragma unroll
  for (int w=0; w<WPT; ++w) {
    const int id = w*get_global_size(0) + get_global_id(0);
    ygm[id] = MultiplyAddVector(ygm[id], alpha, xgm[id]);
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)";

// =================================================================================================
