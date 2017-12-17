
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xgemv kernel (fast versions) for matrix-vector multiplication.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.

// 1: For the full version, see 'xgemv.opencl'

// 2: For the fast version
#ifndef WGS2
  #define WGS2 64     // The local work-group size
#endif
#ifndef WPT2
  #define WPT2 1      // The amount of work-per-thread
#endif
#ifndef VW2
  #define VW2 1       // Vector width of matrix A loads
#endif

// 3: For the fast rotated version
#ifndef WGS3
  #define WGS3 64     // The local work-group size
#endif
#ifndef WPT3
  #define WPT3 1      // The tile-size
#endif
#ifndef VW3
  #define VW3 1       // Vector width of matrix A loads
#endif

// =================================================================================================

// Data-widths for the 'fast' kernel
#if VW2 == 1
  typedef real realVF;
#elif VW2 == 2
  typedef real2 realVF;
#elif VW2 == 4
  typedef real4 realVF;
#elif VW2 == 8
  typedef real8 realVF;
#elif VW2 == 16
  typedef real16 realVF;
#endif

// Data-widths for the 'fast' kernel with rotated matrix
#if VW3 == 1
  typedef real realVFR;
#elif VW3 == 2
  typedef real2 realVFR;
#elif VW3 == 4
  typedef real4 realVFR;
#elif VW3 == 8
  typedef real8 realVFR;
#elif VW3 == 16
  typedef real16 realVFR;
#endif

// =================================================================================================

// Loads a vector input value
INLINE_FUNC realVF LoadMatrixAVF(const __global realVF* restrict agm, const int x, const int y,
                                 const int a_ld) {
  return agm[a_ld*y + x];
}

// =================================================================================================

// Faster version of the kernel, assuming that:
// --> 'm' and 'n' are multiples of WGS2
// --> 'a_offset' is 0
// --> 'a_ld' is a multiple of VW2
// --> 'a_rotated' is 0
// --> 'do_conjugate' is 0
__kernel __attribute__((reqd_work_group_size(WGS2, 1, 1)))
void XgemvFast(const int m, const int n,
               const real_arg arg_alpha,
               const real_arg arg_beta,
               const int a_rotated,
               const __global realVF* restrict agm, const int a_offset, const int a_ld,
               const __global real* restrict xgm, const int x_offset, const int x_inc,
               __global real* ygm, const int y_offset, const int y_inc,
               const int do_conjugate, const int parameter,
               const int kl_unused, const int ku_unused) {
  const real alpha = GetRealArg(arg_alpha);
  const real beta = GetRealArg(arg_beta);

  // Local memory for the vector X
  __local real xlm[WGS2];

  // Initializes the accumulation registers
  #pragma promote_to_registers
  real acc2[WPT2];
  #pragma unroll
  for (int _w = 0; _w < WPT2; _w += 1) {
    SetToZero(acc2[_w]);
  }

  // Loops over work-group sized portions of the work
  for (int kwg=0; kwg<n; kwg+=WGS2) {

    // Loads the vector X into local memory
    const int lid = get_local_id(0);
    xlm[lid] = xgm[(kwg + lid)*x_inc + x_offset];

    // Synchronizes all threads in a workgroup
    barrier(CLK_LOCAL_MEM_FENCE);

    // The multiply-add function (not rotated)
    #pragma unroll
    for (int _kl = 0; _kl < WGS2; _kl += 1) {
      const int k = kwg + _kl;
      #pragma unroll
      for (int _w = 0; _w < WPT2/VW2; _w += 1) {
        const int gid = (WPT2/VW2)*get_global_id(0) + _w;
        realVF avec = agm[(a_ld/VW2)*k + gid];
        #if VW2 == 1
          MultiplyAdd(acc2[VW2*_w+0], xlm[_kl], avec);
        #elif VW2 == 2
          MultiplyAdd(acc2[VW2*_w+0], xlm[_kl], avec.x);
          MultiplyAdd(acc2[VW2*_w+1], xlm[_kl], avec.y);
        #elif VW2 == 4
          MultiplyAdd(acc2[VW2*_w+0], xlm[_kl], avec.x);
          MultiplyAdd(acc2[VW2*_w+1], xlm[_kl], avec.y);
          MultiplyAdd(acc2[VW2*_w+2], xlm[_kl], avec.z);
          MultiplyAdd(acc2[VW2*_w+3], xlm[_kl], avec.w);
        #elif VW2 == 8
          MultiplyAdd(acc2[VW2*_w+0], xlm[_kl], avec.s0);
          MultiplyAdd(acc2[VW2*_w+1], xlm[_kl], avec.s1);
          MultiplyAdd(acc2[VW2*_w+2], xlm[_kl], avec.s2);
          MultiplyAdd(acc2[VW2*_w+3], xlm[_kl], avec.s3);
          MultiplyAdd(acc2[VW2*_w+4], xlm[_kl], avec.s4);
          MultiplyAdd(acc2[VW2*_w+5], xlm[_kl], avec.s5);
          MultiplyAdd(acc2[VW2*_w+6], xlm[_kl], avec.s6);
          MultiplyAdd(acc2[VW2*_w+7], xlm[_kl], avec.s7);
        #elif VW2 == 16
          MultiplyAdd(acc2[VW2*_w+0], xlm[_kl], avec.s0);
          MultiplyAdd(acc2[VW2*_w+1], xlm[_kl], avec.s1);
          MultiplyAdd(acc2[VW2*_w+2], xlm[_kl], avec.s2);
          MultiplyAdd(acc2[VW2*_w+3], xlm[_kl], avec.s3);
          MultiplyAdd(acc2[VW2*_w+4], xlm[_kl], avec.s4);
          MultiplyAdd(acc2[VW2*_w+5], xlm[_kl], avec.s5);
          MultiplyAdd(acc2[VW2*_w+6], xlm[_kl], avec.s6);
          MultiplyAdd(acc2[VW2*_w+7], xlm[_kl], avec.s7);
          MultiplyAdd(acc2[VW2*_w+8], xlm[_kl], avec.s8);
          MultiplyAdd(acc2[VW2*_w+9], xlm[_kl], avec.s9);
          MultiplyAdd(acc2[VW2*_w+10], xlm[_kl], avec.sA);
          MultiplyAdd(acc2[VW2*_w+11], xlm[_kl], avec.sB);
          MultiplyAdd(acc2[VW2*_w+12], xlm[_kl], avec.sC);
          MultiplyAdd(acc2[VW2*_w+13], xlm[_kl], avec.sD);
          MultiplyAdd(acc2[VW2*_w+14], xlm[_kl], avec.sE);
          MultiplyAdd(acc2[VW2*_w+15], xlm[_kl], avec.sF);
        #endif
      }
    }

    // Synchronizes all threads in a workgroup
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Stores the final result
  #pragma unroll
  for (int _w = 0; _w < WPT2; _w += 1) {
    const int gid = WPT2*get_global_id(0) + _w;
    real yval = ygm[gid*y_inc + y_offset];
    AXPBY(ygm[gid*y_inc + y_offset], alpha, acc2[_w], beta, yval);
  }
}

// =================================================================================================

// Faster version of the kernel, assuming that:
// --> 'm' and 'n' are multiples of WGS3
// --> 'a_offset' is 0
// --> 'a_ld' is a multiple of VW3
// --> 'a_rotated' is 1
// --> 'do_conjugate' is 0
__kernel __attribute__((reqd_work_group_size(WGS3, 1, 1)))
void XgemvFastRot(const int m, const int n,
                  const real_arg arg_alpha,
                  const real_arg arg_beta,
                  const int a_rotated,
                  const __global realVFR* restrict agm, const int a_offset, const int a_ld,
                  const __global real* restrict xgm, const int x_offset, const int x_inc,
                  __global real* ygm, const int y_offset, const int y_inc,
                  const int do_conjugate, const int parameter,
                  const int kl_unused, const int ku_unused) {
  const real alpha = GetRealArg(arg_alpha);
  const real beta = GetRealArg(arg_beta);

  // Local memory to store a tile of the matrix (for coalescing)
  __local real tile[WPT3][WGS3];
  const int lid = get_local_id(0);
  const int lid_mod = lid % (WPT3/VW3);
  const int lid_div = lid / (WPT3/VW3);

  // Local memory for the vector X
  __local real xlm[WPT3];

  // Initializes the accumulation register
  real acc3;
  SetToZero(acc3);

  // Loops over tile-sized portions of the work
  for (int kwg=0; kwg<n; kwg+=WPT3) {

    // Loads the vector X into local memory
    if (lid < WPT3) {
      xlm[lid] = xgm[(kwg + lid) * x_inc + x_offset];
    }

    // Loads the matrix A into local memory
    #pragma unroll
    for (int _kl = 0; _kl < WPT3/VW3; _kl += 1) {
      const int x = (kwg/VW3) + lid_mod;
      const int y = get_group_id(0) * WGS3 + lid_div * (WPT3/VW3) + _kl;
      realVFR avec = agm[(a_ld/VW3) * y + x];
      #if VW3 == 1
        tile[_kl*VW3 + 0][lid] = avec;
      #elif VW3 == 2
        tile[_kl*VW3 + 0][lid] = avec.x;
        tile[_kl*VW3 + 1][lid] = avec.y;
      #elif VW3 == 4
        tile[_kl*VW3 + 0][lid] = avec.x;
        tile[_kl*VW3 + 1][lid] = avec.y;
        tile[_kl*VW3 + 2][lid] = avec.z;
        tile[_kl*VW3 + 3][lid] = avec.w;
      #elif VW3 == 8
        tile[_kl*VW3 + 0][lid] = avec.s0;
        tile[_kl*VW3 + 1][lid] = avec.s1;
        tile[_kl*VW3 + 2][lid] = avec.s2;
        tile[_kl*VW3 + 3][lid] = avec.s3;
        tile[_kl*VW3 + 4][lid] = avec.s4;
        tile[_kl*VW3 + 5][lid] = avec.s5;
        tile[_kl*VW3 + 6][lid] = avec.s6;
        tile[_kl*VW3 + 7][lid] = avec.s7;
      #elif VW3 == 16
        tile[_kl*VW3 + 0][lid] = avec.s0;
        tile[_kl*VW3 + 1][lid] = avec.s1;
        tile[_kl*VW3 + 2][lid] = avec.s2;
        tile[_kl*VW3 + 3][lid] = avec.s3;
        tile[_kl*VW3 + 4][lid] = avec.s4;
        tile[_kl*VW3 + 5][lid] = avec.s5;
        tile[_kl*VW3 + 6][lid] = avec.s6;
        tile[_kl*VW3 + 7][lid] = avec.s7;
        tile[_kl*VW3 + 8][lid] = avec.s8;
        tile[_kl*VW3 + 9][lid] = avec.s9;
        tile[_kl*VW3 + 10][lid] = avec.sA;
        tile[_kl*VW3 + 11][lid] = avec.sB;
        tile[_kl*VW3 + 12][lid] = avec.sC;
        tile[_kl*VW3 + 13][lid] = avec.sD;
        tile[_kl*VW3 + 14][lid] = avec.sE;
        tile[_kl*VW3 + 15][lid] = avec.sF;
      #endif
    }

    // Synchronizes all threads in a workgroup
    barrier(CLK_LOCAL_MEM_FENCE);

    // The multiply-add function (rotated)
    #pragma unroll
    for (int _kl = 0; _kl < WPT3/VW3; _kl += 1) {
      #pragma unroll
      for (int _v = 0; _v < VW3; _v += 1) {
        real aval = tile[lid_mod*VW3 + _v][lid_div * (WPT3/VW3) + _kl];
        real xval = xlm[_kl*VW3 + _v];
        MultiplyAdd(acc3, xval, aval);
      }
    }

    // Synchronizes all threads in a workgroup
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Stores the final result
  const int gid = get_global_id(0);
  real yval = ygm[gid * y_inc + y_offset];
  AXPBY(ygm[gid * y_inc + y_offset], alpha, acc3, beta, yval);
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
