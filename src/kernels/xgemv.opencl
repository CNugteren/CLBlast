
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xgemv kernel for matrix-vector multiplication.
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
  #define VW 1       // Vector width of matrix A loads (only for the fast kernel)
#endif

// =================================================================================================

// Full version of the kernel
__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void Xgemv(const int m, const int n, const real alpha, const real beta,
                    const int a_transposed,
                    const __global real* restrict agm, const int a_offset, const int a_ld,
                    const __global real* restrict xgm, const int x_offset, const int x_inc,
                    __global real* ygm, const int y_offset, const int y_inc) {

  // Local memory for the vector X
  __local real xlm[WGS];

  // Initializes the accumulation register
  real acc[WPT];
  #pragma unroll
  for (int w=0; w<WPT; ++w) {
    SetToZero(acc[w]);
  }

  // Divides the work in a main and tail section
  const int n_tail = n % WGS;
  const int n_floor = n - n_tail;

  // Loops over work-group sized portions of the work
  for (int kwg=0; kwg<n_floor; kwg+=WGS) {

    // Loads the vector X into local memory
    const int lid = get_local_id(0);
    xlm[lid] = xgm[(kwg + lid)*x_inc + x_offset];

    // Synchronizes all threads in a workgroup
    barrier(CLK_LOCAL_MEM_FENCE);

    // Loops over the work per thread, and checks whether in bounds
    #pragma unroll
    for (int w=0; w<WPT; ++w) {
      const int gid = w*get_global_size(0) + get_global_id(0);
      if (gid < m) {

        // The multiply-add function for the main part (divisable by WGS)
        if (a_transposed == 0) { // Not transposed
          #pragma unroll
          for (int kl=0; kl<WGS; ++kl) {
            const int k = kwg + kl;
            MultiplyAdd(acc[w], xlm[kl], agm[gid + a_ld*k + a_offset]);
          }
        }
        else { // Transposed
          #pragma unroll
          for (int kl=0; kl<WGS; ++kl) {
            const int k = kwg + kl;
            MultiplyAdd(acc[w], xlm[kl], agm[k + a_ld*gid + a_offset]);
          }
        }
      }
    }

    // Synchronizes all threads in a workgroup
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Loops over the work per thread, and checks whether in bounds
  #pragma unroll
  for (int w=0; w<WPT; ++w) {
    const int gid = w*get_global_size(0) + get_global_id(0);
    if (gid < m) {

      // The multiply-add function for the remainder part (not divisable by WGS)
      if (a_transposed == 0) { // Not transposed
        #pragma unroll
        for (int k=n_floor; k<n; ++k) {
          MultiplyAdd(acc[w], xgm[k*x_inc + x_offset], agm[gid + a_ld*k + a_offset]);
        }
      }
      else { // Transposed
        #pragma unroll
        for (int k=n_floor; k<n; ++k) {
          MultiplyAdd(acc[w], xgm[k*x_inc + x_offset], agm[k + a_ld*gid + a_offset]);
        }
      }

      // Stores the final result
      AXPBY(ygm[gid*y_inc + y_offset], alpha, acc[w], beta, ygm[gid*y_inc + y_offset]);
    }
  }
}

// =================================================================================================

// Data-widths for the 'fast' kernel
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

// Faster version of the kernel, assuming that:
// --> 'm' and 'n' are multiples of WGS
// --> 'a_offset' is 0
// --> 'a_ld' is a multiple of VW
__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void XgemvFast(const int m, const int n, const real alpha, const real beta,
                        const int a_transposed,
                        const __global realV* restrict agm, const int a_offset, const int a_ld,
                        const __global real* restrict xgm, const int x_offset, const int x_inc,
                        __global real* ygm, const int y_offset, const int y_inc) {
  // Local memory for the vector X
  __local real xlm[WGS];

  // Initializes the accumulation register
  real acc[WPT];
  #pragma unroll
  for (int w=0; w<WPT; ++w) {
    SetToZero(acc[w]);
  }

  // Loops over work-group sized portions of the work
  for (int kwg=0; kwg<n; kwg+=WGS) {

    // Loads the vector X into local memory
    const int lid = get_local_id(0);
    xlm[lid] = xgm[(kwg + lid)*x_inc + x_offset];

    // Synchronizes all threads in a workgroup
    barrier(CLK_LOCAL_MEM_FENCE);

    // The multiply-add function (not transposed)
    if (a_transposed == 0) {
      #pragma unroll
      for (int kl=0; kl<WGS; ++kl) {
        const int k = kwg + kl;
        #pragma unroll
        for (int w=0; w<WPT/VW; ++w) {
          const int gid = (WPT/VW)*get_global_id(0) + w;
          #if VW == 1
            MultiplyAdd(acc[VW*w+0], xlm[kl], agm[gid + (a_ld/VW)*k]);
          #elif VW == 2
            MultiplyAdd(acc[VW*w+0], xlm[kl], agm[gid + (a_ld/VW)*k].x);
            MultiplyAdd(acc[VW*w+1], xlm[kl], agm[gid + (a_ld/VW)*k].y);
          #elif VW == 4
            MultiplyAdd(acc[VW*w+0], xlm[kl], agm[gid + (a_ld/VW)*k].x);
            MultiplyAdd(acc[VW*w+1], xlm[kl], agm[gid + (a_ld/VW)*k].y);
            MultiplyAdd(acc[VW*w+2], xlm[kl], agm[gid + (a_ld/VW)*k].z);
            MultiplyAdd(acc[VW*w+3], xlm[kl], agm[gid + (a_ld/VW)*k].w);
          #elif VW == 8
            MultiplyAdd(acc[VW*w+0], xlm[kl], agm[gid + (a_ld/VW)*k].s0);
            MultiplyAdd(acc[VW*w+1], xlm[kl], agm[gid + (a_ld/VW)*k].s1);
            MultiplyAdd(acc[VW*w+2], xlm[kl], agm[gid + (a_ld/VW)*k].s2);
            MultiplyAdd(acc[VW*w+3], xlm[kl], agm[gid + (a_ld/VW)*k].s3);
            MultiplyAdd(acc[VW*w+4], xlm[kl], agm[gid + (a_ld/VW)*k].s4);
            MultiplyAdd(acc[VW*w+5], xlm[kl], agm[gid + (a_ld/VW)*k].s5);
            MultiplyAdd(acc[VW*w+6], xlm[kl], agm[gid + (a_ld/VW)*k].s6);
            MultiplyAdd(acc[VW*w+7], xlm[kl], agm[gid + (a_ld/VW)*k].s7);
          #elif VW == 16
            MultiplyAdd(acc[VW*w+0], xlm[kl], agm[gid + (a_ld/VW)*k].s0);
            MultiplyAdd(acc[VW*w+1], xlm[kl], agm[gid + (a_ld/VW)*k].s1);
            MultiplyAdd(acc[VW*w+2], xlm[kl], agm[gid + (a_ld/VW)*k].s2);
            MultiplyAdd(acc[VW*w+3], xlm[kl], agm[gid + (a_ld/VW)*k].s3);
            MultiplyAdd(acc[VW*w+4], xlm[kl], agm[gid + (a_ld/VW)*k].s4);
            MultiplyAdd(acc[VW*w+5], xlm[kl], agm[gid + (a_ld/VW)*k].s5);
            MultiplyAdd(acc[VW*w+6], xlm[kl], agm[gid + (a_ld/VW)*k].s6);
            MultiplyAdd(acc[VW*w+7], xlm[kl], agm[gid + (a_ld/VW)*k].s7);
            MultiplyAdd(acc[VW*w+8], xlm[kl], agm[gid + (a_ld/VW)*k].s8);
            MultiplyAdd(acc[VW*w+9], xlm[kl], agm[gid + (a_ld/VW)*k].s9);
            MultiplyAdd(acc[VW*w+10], xlm[kl], agm[gid + (a_ld/VW)*k].sA);
            MultiplyAdd(acc[VW*w+11], xlm[kl], agm[gid + (a_ld/VW)*k].sB);
            MultiplyAdd(acc[VW*w+12], xlm[kl], agm[gid + (a_ld/VW)*k].sC);
            MultiplyAdd(acc[VW*w+13], xlm[kl], agm[gid + (a_ld/VW)*k].sD);
            MultiplyAdd(acc[VW*w+14], xlm[kl], agm[gid + (a_ld/VW)*k].sE);
            MultiplyAdd(acc[VW*w+15], xlm[kl], agm[gid + (a_ld/VW)*k].sF);
          #endif
        }
      }
    }

    // The multiply-add function (transposed)
    else {
      #pragma unroll
      for (int kl=0; kl<WGS/VW; ++kl) {
        const int k = (kwg/VW) + kl;
        #pragma unroll
        for (int w=0; w<WPT; ++w) {
          const int gid = WPT*get_global_id(0) + w;
          realV avec = agm[k + (a_ld/VW)*gid];
          #if VW == 1
            MultiplyAdd(acc[w], xlm[VW*kl+0], avec);
          #elif VW == 2
            MultiplyAdd(acc[w], xlm[VW*kl+0], avec.x);
            MultiplyAdd(acc[w], xlm[VW*kl+1], avec.y);
          #elif VW == 4
            MultiplyAdd(acc[w], xlm[VW*kl+0], avec.x);
            MultiplyAdd(acc[w], xlm[VW*kl+1], avec.y);
            MultiplyAdd(acc[w], xlm[VW*kl+2], avec.z);
            MultiplyAdd(acc[w], xlm[VW*kl+3], avec.w);
          #elif VW == 8
            MultiplyAdd(acc[w], xlm[VW*kl+0], avec.s0);
            MultiplyAdd(acc[w], xlm[VW*kl+1], avec.s1);
            MultiplyAdd(acc[w], xlm[VW*kl+2], avec.s2);
            MultiplyAdd(acc[w], xlm[VW*kl+3], avec.s3);
            MultiplyAdd(acc[w], xlm[VW*kl+4], avec.s4);
            MultiplyAdd(acc[w], xlm[VW*kl+5], avec.s5);
            MultiplyAdd(acc[w], xlm[VW*kl+6], avec.s6);
            MultiplyAdd(acc[w], xlm[VW*kl+7], avec.s7);
          #elif VW == 16
            MultiplyAdd(acc[w], xlm[VW*kl+0], avec.s0);
            MultiplyAdd(acc[w], xlm[VW*kl+1], avec.s1);
            MultiplyAdd(acc[w], xlm[VW*kl+2], avec.s2);
            MultiplyAdd(acc[w], xlm[VW*kl+3], avec.s3);
            MultiplyAdd(acc[w], xlm[VW*kl+4], avec.s4);
            MultiplyAdd(acc[w], xlm[VW*kl+5], avec.s5);
            MultiplyAdd(acc[w], xlm[VW*kl+6], avec.s6);
            MultiplyAdd(acc[w], xlm[VW*kl+7], avec.s7);
            MultiplyAdd(acc[w], xlm[VW*kl+8], avec.s8);
            MultiplyAdd(acc[w], xlm[VW*kl+9], avec.s9);
            MultiplyAdd(acc[w], xlm[VW*kl+10], avec.sA);
            MultiplyAdd(acc[w], xlm[VW*kl+11], avec.sB);
            MultiplyAdd(acc[w], xlm[VW*kl+12], avec.sC);
            MultiplyAdd(acc[w], xlm[VW*kl+13], avec.sD);
            MultiplyAdd(acc[w], xlm[VW*kl+14], avec.sE);
            MultiplyAdd(acc[w], xlm[VW*kl+15], avec.sF);
          #endif
        }
      }
    }

    // Synchronizes all threads in a workgroup
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Stores the final result
  #pragma unroll
  for (int w=0; w<WPT; ++w) {
    const int gid = WPT*get_global_id(0) + w;
    AXPBY(ygm[gid*y_inc + y_offset], alpha, acc[w], beta, ygm[gid*y_inc + y_offset]);
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)";

// =================================================================================================
