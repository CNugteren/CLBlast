
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
  #define WPT3 1      // The amount of work-per-thread
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

// Loads a vector input value (1/2)
inline realVF LoadMatrixAVF(const __global realVF* restrict agm, const int x, const int y,
                            const int a_ld) {
  return agm[a_ld*y + x];
}

// Loads a vector input value (2/2): as before, but different data-type
inline realVFR LoadMatrixAVFR(const __global realVFR* restrict agm, const int x, const int y,
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
__attribute__((reqd_work_group_size(WGS2, 1, 1)))
__kernel void XgemvFast(const int m, const int n,
                        const real_arg arg_alpha,
                        const real_arg arg_beta,
                        const int a_rotated,
                        const __global realVF* restrict agm, const int a_offset, const int a_ld,
                        const __global real* restrict xgm, const int x_offset, const int x_inc,
                        __global real* ygm, const int y_offset, const int y_inc,
                        const int do_conjugate, const int parameter,
                        const int kl, const int ku) {
  const real alpha = GetRealArg(arg_alpha);
  const real beta = GetRealArg(arg_beta);

  // Local memory for the vector X
  __local real xlm[WGS2];

  // Initializes the accumulation register
  real acc[WPT2];
  #pragma unroll
  for (int w=0; w<WPT2; ++w) {
    SetToZero(acc[w]);
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
    for (int kl=0; kl<WGS2; ++kl) {
      const int k = kwg + kl;
      #pragma unroll
      for (int w=0; w<WPT2/VW2; ++w) {
        const int gid = (WPT2/VW2)*get_global_id(0) + w;
        realVF avec = LoadMatrixAVF(agm, gid, k, a_ld/VW2);
        #if VW2 == 1
          MultiplyAdd(acc[VW2*w+0], xlm[kl], avec);
        #elif VW2 == 2
          MultiplyAdd(acc[VW2*w+0], xlm[kl], avec.x);
          MultiplyAdd(acc[VW2*w+1], xlm[kl], avec.y);
        #elif VW2 == 4
          MultiplyAdd(acc[VW2*w+0], xlm[kl], avec.x);
          MultiplyAdd(acc[VW2*w+1], xlm[kl], avec.y);
          MultiplyAdd(acc[VW2*w+2], xlm[kl], avec.z);
          MultiplyAdd(acc[VW2*w+3], xlm[kl], avec.w);
        #elif VW2 == 8
          MultiplyAdd(acc[VW2*w+0], xlm[kl], avec.s0);
          MultiplyAdd(acc[VW2*w+1], xlm[kl], avec.s1);
          MultiplyAdd(acc[VW2*w+2], xlm[kl], avec.s2);
          MultiplyAdd(acc[VW2*w+3], xlm[kl], avec.s3);
          MultiplyAdd(acc[VW2*w+4], xlm[kl], avec.s4);
          MultiplyAdd(acc[VW2*w+5], xlm[kl], avec.s5);
          MultiplyAdd(acc[VW2*w+6], xlm[kl], avec.s6);
          MultiplyAdd(acc[VW2*w+7], xlm[kl], avec.s7);
        #elif VW2 == 16
          MultiplyAdd(acc[VW2*w+0], xlm[kl], avec.s0);
          MultiplyAdd(acc[VW2*w+1], xlm[kl], avec.s1);
          MultiplyAdd(acc[VW2*w+2], xlm[kl], avec.s2);
          MultiplyAdd(acc[VW2*w+3], xlm[kl], avec.s3);
          MultiplyAdd(acc[VW2*w+4], xlm[kl], avec.s4);
          MultiplyAdd(acc[VW2*w+5], xlm[kl], avec.s5);
          MultiplyAdd(acc[VW2*w+6], xlm[kl], avec.s6);
          MultiplyAdd(acc[VW2*w+7], xlm[kl], avec.s7);
          MultiplyAdd(acc[VW2*w+8], xlm[kl], avec.s8);
          MultiplyAdd(acc[VW2*w+9], xlm[kl], avec.s9);
          MultiplyAdd(acc[VW2*w+10], xlm[kl], avec.sA);
          MultiplyAdd(acc[VW2*w+11], xlm[kl], avec.sB);
          MultiplyAdd(acc[VW2*w+12], xlm[kl], avec.sC);
          MultiplyAdd(acc[VW2*w+13], xlm[kl], avec.sD);
          MultiplyAdd(acc[VW2*w+14], xlm[kl], avec.sE);
          MultiplyAdd(acc[VW2*w+15], xlm[kl], avec.sF);
        #endif
      }
    }

    // Synchronizes all threads in a workgroup
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Stores the final result
  #pragma unroll
  for (int w=0; w<WPT2; ++w) {
    const int gid = WPT2*get_global_id(0) + w;
    real yval = ygm[gid*y_inc + y_offset];
    AXPBY(ygm[gid*y_inc + y_offset], alpha, acc[w], beta, yval);
  }
}

// =================================================================================================

// Faster version of the kernel, assuming that:
// --> 'm' and 'n' are multiples of WGS3
// --> 'a_offset' is 0
// --> 'a_ld' is a multiple of VW3
// --> 'a_rotated' is 1
// --> 'do_conjugate' is 0
__attribute__((reqd_work_group_size(WGS3, 1, 1)))
__kernel void XgemvFastRot(const int m, const int n,
                           const real_arg arg_alpha,
                           const real_arg arg_beta,
                           const int a_rotated,
                           const __global realVFR* restrict agm, const int a_offset, const int a_ld,
                           const __global real* restrict xgm, const int x_offset, const int x_inc,
                           __global real* ygm, const int y_offset, const int y_inc,
                           const int do_conjugate, const int parameter,
                           const int kl, const int ku) {
  const real alpha = GetRealArg(arg_alpha);
  const real beta = GetRealArg(arg_beta);

  // Local memory for the vector X
  __local real xlm[WGS3];

  // Initializes the accumulation register
  real acc[WPT3];
  #pragma unroll
  for (int w=0; w<WPT3; ++w) {
    SetToZero(acc[w]);
  }

  // Loops over work-group sized portions of the work
  for (int kwg=0; kwg<n; kwg+=WGS3) {

    // Loads the vector X into local memory
    const int lid = get_local_id(0);
    xlm[lid] = xgm[(kwg + lid)*x_inc + x_offset];

    // Synchronizes all threads in a workgroup
    barrier(CLK_LOCAL_MEM_FENCE);

    // The multiply-add function (rotated)
    #pragma unroll
    for (int kl=0; kl<WGS3/VW3; ++kl) {
      const int k = (kwg/VW3) + kl;
      #pragma unroll
      for (int w=0; w<WPT3; ++w) {
        const int gid = WPT3*get_global_id(0) + w;
        realVFR avec = LoadMatrixAVFR(agm, k, gid, a_ld/VW3);
        #if VW3 == 1
          MultiplyAdd(acc[w], xlm[VW3*kl+0], avec);
        #elif VW3 == 2
          MultiplyAdd(acc[w], xlm[VW3*kl+0], avec.x);
          MultiplyAdd(acc[w], xlm[VW3*kl+1], avec.y);
        #elif VW3 == 4
          MultiplyAdd(acc[w], xlm[VW3*kl+0], avec.x);
          MultiplyAdd(acc[w], xlm[VW3*kl+1], avec.y);
          MultiplyAdd(acc[w], xlm[VW3*kl+2], avec.z);
          MultiplyAdd(acc[w], xlm[VW3*kl+3], avec.w);
        #elif VW3 == 8
          MultiplyAdd(acc[w], xlm[VW3*kl+0], avec.s0);
          MultiplyAdd(acc[w], xlm[VW3*kl+1], avec.s1);
          MultiplyAdd(acc[w], xlm[VW3*kl+2], avec.s2);
          MultiplyAdd(acc[w], xlm[VW3*kl+3], avec.s3);
          MultiplyAdd(acc[w], xlm[VW3*kl+4], avec.s4);
          MultiplyAdd(acc[w], xlm[VW3*kl+5], avec.s5);
          MultiplyAdd(acc[w], xlm[VW3*kl+6], avec.s6);
          MultiplyAdd(acc[w], xlm[VW3*kl+7], avec.s7);
        #elif VW3 == 16
          MultiplyAdd(acc[w], xlm[VW3*kl+0], avec.s0);
          MultiplyAdd(acc[w], xlm[VW3*kl+1], avec.s1);
          MultiplyAdd(acc[w], xlm[VW3*kl+2], avec.s2);
          MultiplyAdd(acc[w], xlm[VW3*kl+3], avec.s3);
          MultiplyAdd(acc[w], xlm[VW3*kl+4], avec.s4);
          MultiplyAdd(acc[w], xlm[VW3*kl+5], avec.s5);
          MultiplyAdd(acc[w], xlm[VW3*kl+6], avec.s6);
          MultiplyAdd(acc[w], xlm[VW3*kl+7], avec.s7);
          MultiplyAdd(acc[w], xlm[VW3*kl+8], avec.s8);
          MultiplyAdd(acc[w], xlm[VW3*kl+9], avec.s9);
          MultiplyAdd(acc[w], xlm[VW3*kl+10], avec.sA);
          MultiplyAdd(acc[w], xlm[VW3*kl+11], avec.sB);
          MultiplyAdd(acc[w], xlm[VW3*kl+12], avec.sC);
          MultiplyAdd(acc[w], xlm[VW3*kl+13], avec.sD);
          MultiplyAdd(acc[w], xlm[VW3*kl+14], avec.sE);
          MultiplyAdd(acc[w], xlm[VW3*kl+15], avec.sF);
        #endif
      }
    }

    // Synchronizes all threads in a workgroup
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Stores the final result
  #pragma unroll
  for (int w=0; w<WPT3; ++w) {
    const int gid = WPT3*get_global_id(0) + w;
    real yval = ygm[gid*y_inc + y_offset];
    AXPBY(ygm[gid*y_inc + y_offset], alpha, acc[w], beta, yval);
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
