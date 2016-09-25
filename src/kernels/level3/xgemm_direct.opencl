
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This is a generic GEMM kernel that works for all sizes and configurations: it doesn't require any
// pre and and post-processing kernels.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library. Note that all parameters here have a
// suffix 'D' to denote that they are for the 'direct' version of the GEMM kernel.
#ifndef MWGD
  #define MWGD 8      // Tile-size in dimension M (e.g. 64, 128)
#endif
#ifndef NWGD
  #define NWGD 8      // Tile-size in dimension N (e.g. 64, 128)
#endif
#ifndef KWGD
  #define KWGD 8      // Tile-size in dimension K (e.g. 8, 16)
#endif
#ifndef MDIMCD
  #define MDIMCD 8    // Threads per workgroup in M-dimension (e.g. 8, 16, 32)
#endif
#ifndef NDIMCD
  #define NDIMCD 8    // Threads per workgroup in N-dimension (e.g. 8, 16, 32)
#endif
#ifndef MDIMAD
  #define MDIMAD 8    // Re-shaped tile dimension of matrix A: KDIMAD * MDIMAD
#endif
#ifndef NDIMBD
  #define NDIMBD 8    // Re-shaped tile dimension of matrix B: KDIMBD * NDIMBD
#endif
#ifndef KWID
  #define KWID 1      // Unroll factor of the KWGD loop (smaller or equal than KWGD)
#endif
#ifndef VWMD
  #define VWMD 1      // Vector width of matrices A and C
#endif
#ifndef VWND
  #define VWND 1      // Vector width of matrix B
#endif

// Helper parameters based on the above tuning parameters
#define MWID (MWGD/MDIMCD)                // Work per work-item (M-dimension)
#define NWID (NWGD/NDIMCD)                // Work per work-item (N-dimension)
#define KDIMAD ((MDIMCD*NDIMCD)/(MDIMAD)) // Re-shaped tile dimension of matrix A: KDIMAD * MDIMAD
#define KDIMBD ((MDIMCD*NDIMCD)/(NDIMBD)) // Re-shaped tile dimension of matrix B: KDIMBD * NDIMBD
#define MWAD (MWGD/MDIMAD)                // Amount of loads-per-thread for matrix A (M-dimension)
#define KWAD (KWGD/KDIMAD)                // Amount of loads-per-thread for matrix A (K-dimension)
#define KWBD (KWGD/KDIMBD)                // Amount of loads-per-thread for matrix B (K-dimension)
#define NWBD (NWGD/NDIMBD)                // Amount of loads-per-thread for matrix B (N-dimension)

// =================================================================================================

// Data-widths in dimension M
#if VWMD == 1
    typedef real realMD;
#elif VWMD == 2
    typedef real2 realMD;
#elif VWMD == 4
    typedef real4 realMD;
#elif VWMD == 8
    typedef real8 realMD;
#elif VWMD == 16
    typedef real16 realMD;
#endif

// Data-widths in dimension N
#if VWND == 1
    typedef real realND;
#elif VWND == 2
    typedef real2 realND;
#elif VWND == 4
    typedef real4 realND;
#elif VWND == 8
    typedef real8 realND;
#elif VWND == 16
    typedef real16 realND;
#endif

// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix.
inline void GlobalToLocalDirectA(const __global realMD* restrict agm, __local real* alm,
                                 const int a_ld, const int a_offset, const int tid, const int kwg,
                                 const int a_transpose, const int a_conjugate) {
  const int la0 = tid % MDIMAD;
  const int la1 = tid / MDIMAD;
  #pragma unroll
  for (int mia=0; mia<MWAD/VWMD; ++mia) {
    #pragma unroll
    for (int kia=0; kia<KWAD; ++kia) {

      // Computes the indices for the global memory
      int mg = mia + la0*(MWAD/VWMD);
      int kg = kia + la1*KWAD;
      int idm = (a_transpose) ? mg + kwg/VWMD : mg + GetGroupID0()*(MWGD/VWMD);
      int idk = (a_transpose) ? kg + GetGroupID0()*MWGD : kg + kwg;

      // Loads the data from global memory into the local memory
      const realMD avec = agm[idk*(a_ld/VWMD) + idm + a_offset];
      #if VWMD == 1
         alm[kg*MWGD + mg] = avec;
      #elif VWMD == 2
         alm[kg*MWGD + mg*VWMD + 0] = avec.x;
         alm[kg*MWGD + mg*VWMD + 1] = avec.y;
      #elif VWMD == 4
         alm[kg*MWGD + mg*VWMD + 0] = avec.x;
         alm[kg*MWGD + mg*VWMD + 1] = avec.y;
         alm[kg*MWGD + mg*VWMD + 2] = avec.z;
         alm[kg*MWGD + mg*VWMD + 3] = avec.w;
      #elif VWMD == 8
         alm[kg*MWGD + mg*VWMD + 0] = avec.s0;
         alm[kg*MWGD + mg*VWMD + 1] = avec.s1;
         alm[kg*MWGD + mg*VWMD + 2] = avec.s2;
         alm[kg*MWGD + mg*VWMD + 3] = avec.s3;
         alm[kg*MWGD + mg*VWMD + 4] = avec.s4;
         alm[kg*MWGD + mg*VWMD + 5] = avec.s5;
         alm[kg*MWGD + mg*VWMD + 6] = avec.s6;
         alm[kg*MWGD + mg*VWMD + 7] = avec.s7;
      #elif VWMD == 16
         alm[kg*MWGD + mg*VWMD + 0] = avec.s0;
         alm[kg*MWGD + mg*VWMD + 1] = avec.s1;
         alm[kg*MWGD + mg*VWMD + 2] = avec.s2;
         alm[kg*MWGD + mg*VWMD + 3] = avec.s3;
         alm[kg*MWGD + mg*VWMD + 4] = avec.s4;
         alm[kg*MWGD + mg*VWMD + 5] = avec.s5;
         alm[kg*MWGD + mg*VWMD + 6] = avec.s6;
         alm[kg*MWGD + mg*VWMD + 7] = avec.s7;
         alm[kg*MWGD + mg*VWMD + 8] = avec.s8;
         alm[kg*MWGD + mg*VWMD + 9] = avec.s9;
         alm[kg*MWGD + mg*VWMD + 10] = avec.sA;
         alm[kg*MWGD + mg*VWMD + 11] = avec.sB;
         alm[kg*MWGD + mg*VWMD + 12] = avec.sC;
         alm[kg*MWGD + mg*VWMD + 13] = avec.sD;
         alm[kg*MWGD + mg*VWMD + 14] = avec.sE;
         alm[kg*MWGD + mg*VWMD + 15] = avec.sF;
      #endif
      if (a_conjugate) {
        for (int vm=0; vm<VWMD; ++vm) {
          COMPLEX_CONJUGATE(alm[kg*MWGD + mg*VWMD + vm]);
        }
      }
    }
  }
}

// Same as above, but now for the B input matrix
inline void GlobalToLocalDirectB(const __global realND* restrict bgm, __local real* blm,
                                 const int b_ld, const int b_offset, const int tid, const int kwg,
                                 const int b_transpose, const int b_conjugate) {
  const int lb0 = tid % NDIMBD;
  const int lb1 = tid / NDIMBD;
  #pragma unroll
  for (int kib=0; kib<KWBD; ++kib) {
    #pragma unroll
    for (int nib=0; nib<NWBD/VWND; ++nib) {

      // Computes the indices for the global memory
      int ng = nib + lb0*(NWBD/VWND);
      int kg = kib + lb1*KWBD;
      int idn = (b_transpose) ? ng + kwg/VWND : ng + GetGroupID1()*(NWGD/VWND);
      int idk = (b_transpose) ? kg + GetGroupID1()*NWGD : kg + kwg;

      // Loads the data from global memory into the local memory
      const realMD bvec = bgm[idk*(b_ld/VWND) + idn + b_offset];
      #if VWND == 1
         blm[kg*NWGD + ng] = bvec;
      #elif VWND == 2
         blm[kg*NWGD + ng*VWND + 0] = bvec.x;
         blm[kg*NWGD + ng*VWND + 1] = bvec.y;
      #elif VWND == 4
         blm[kg*NWGD + ng*VWND + 0] = bvec.x;
         blm[kg*NWGD + ng*VWND + 1] = bvec.y;
         blm[kg*NWGD + ng*VWND + 2] = bvec.z;
         blm[kg*NWGD + ng*VWND + 3] = bvec.w;
      #elif VWND == 8
         blm[kg*NWGD + ng*VWND + 0] = bvec.s0;
         blm[kg*NWGD + ng*VWND + 1] = bvec.s1;
         blm[kg*NWGD + ng*VWND + 2] = bvec.s2;
         blm[kg*NWGD + ng*VWND + 3] = bvec.s3;
         blm[kg*NWGD + ng*VWND + 4] = bvec.s4;
         blm[kg*NWGD + ng*VWND + 5] = bvec.s5;
         blm[kg*NWGD + ng*VWND + 6] = bvec.s6;
         blm[kg*NWGD + ng*VWND + 7] = bvec.s7;
      #elif VWND == 16
         blm[kg*NWGD + ng*VWND + 0] = bvec.s0;
         blm[kg*NWGD + ng*VWND + 1] = bvec.s1;
         blm[kg*NWGD + ng*VWND + 2] = bvec.s2;
         blm[kg*NWGD + ng*VWND + 3] = bvec.s3;
         blm[kg*NWGD + ng*VWND + 4] = bvec.s4;
         blm[kg*NWGD + ng*VWND + 5] = bvec.s5;
         blm[kg*NWGD + ng*VWND + 6] = bvec.s6;
         blm[kg*NWGD + ng*VWND + 7] = bvec.s7;
         blm[kg*NWGD + ng*VWND + 8] = bvec.s8;
         blm[kg*NWGD + ng*VWND + 9] = bvec.s9;
         blm[kg*NWGD + ng*VWND + 10] = bvec.sA;
         blm[kg*NWGD + ng*VWND + 11] = bvec.sB;
         blm[kg*NWGD + ng*VWND + 12] = bvec.sC;
         blm[kg*NWGD + ng*VWND + 13] = bvec.sD;
         blm[kg*NWGD + ng*VWND + 14] = bvec.sE;
         blm[kg*NWGD + ng*VWND + 15] = bvec.sF;
      #endif
      if (b_conjugate) {
        for (int vn=0; vn<VWND; ++vn) {
          COMPLEX_CONJUGATE(blm[kg*NWGD + ng*VWND + vn]);
        }
      }
    }
  }
}

// =================================================================================================

// Caches on-chip local memory into per-thread private memory (registers). This function is specific
// for caching the A input matrix.
inline void LocalToPrivateDirectA(__local real* alm, real apm[MWID], const int kg,
                                  const int a_transpose) {
  #pragma unroll
  for (int mi=0; mi<MWID; ++mi) {
    const int mg = mi + get_local_id(0)*MWID;
    const int index = (a_transpose) ? mg*KWGD + kg : kg*MWGD + mg;
    apm[mi] = alm[index];
  }
}

// Same as above, but now for the B input matrix
inline void LocalToPrivateDirectB(__local real* blm, real bpm[NWID], const int kg,
                                  const int b_transpose) {
  #pragma unroll
  for (int ni=0; ni<NWID; ++ni) {
    const int ng = ni + get_local_id(1)*NWID;
    const int index = (b_transpose) ? ng*KWGD + kg : kg*NWGD + ng;
    bpm[ni] = blm[index];
  }
}

// =================================================================================================

// Initializes the accumulation registers to zero
inline void InitAccRegistersDirect(real cpm[NWID][MWID]) {
  #pragma unroll
  for (int mi=0; mi<MWID; ++mi) {
    #pragma unroll
    for (int ni=0; ni<NWID; ++ni) {
      SetToZero(cpm[ni][mi]);
    }
  }
}

// =================================================================================================

// Performs the actual computation: Cpm += Apm * Bpm
inline void MultiplyAccumulateDirect(real cpm[NWID][MWID], real apm[MWID], real bpm[NWID]) {
  #pragma unroll
  for (int ni=0; ni<NWID; ++ni) {
    #pragma unroll
    for (int mi=0; mi<MWID; ++mi) {
      MultiplyAdd(cpm[ni][mi], apm[mi], bpm[ni]);
    }
  }
}

// =================================================================================================

// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication
// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm
inline void StoreResultsDirect(__global real* cgm, real cpm[NWID][MWID],
                               const int kSizeM, const int kSizeN,
                               const real alpha, const real beta,
                               const int c_ld, const int c_offset, const int c_transpose) {
  #pragma unroll
  for (int ni=0; ni<NWID; ++ni) {
    #pragma unroll
    for (int mi=0; mi<MWID; ++mi) {
      int mg = mi + get_local_id(0)*MWID;
      int ng = ni + get_local_id(1)*NWID;
      int idm = mg + GetGroupID0() * MWGD;
      int idn = ng + GetGroupID1() * NWGD;

      // Determines the destination index
      const int c_index = (c_transpose) ? idm*c_ld + idn : idn*c_ld + idm;

      // The final multiplication with alpha and the addition with beta*C
      real result;
      AXPBY(result, alpha, cpm[ni][mi], beta, cgm[c_index + c_offset]);
      cgm[c_index + c_offset] = result;
    }
  }
}

// =================================================================================================

// Main entry point of the kernel. This is the direct version without restrictions.
__attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
__kernel void XgemmDirect(const int kSizeM, const int kSizeN, const int kSizeK,
                          const real_arg arg_alpha,
                          const real_arg arg_beta,
                          const __global realMD* restrict agm, const int a_offset, const int a_ld,
                          const __global realND* restrict bgm, const int b_offset, const int b_ld,
                          __global real* cgm, const int c_offset, const int c_ld,
                          const int a_transpose, const int b_transpose, const int c_transpose,
                          const int a_conjugate, const int b_conjugate) {
  const real alpha = GetRealArg(arg_alpha);
  const real beta = GetRealArg(arg_beta);

  // Extra pointers to scalar versions of global memory
  const __global real* restrict agms = (const __global real* restrict) agm;
  const __global real* restrict bgms = (const __global real* restrict) bgm;

  // Allocates workgroup-private memory (local memory)
  __local real alm[KWGD * MWGD];
  __local real blm[KWGD * NWGD];

  // Combined thread identifier (volatile to disable caching)
  volatile int tid = get_local_id(0) + MDIMCD*get_local_id(1);

  // Allocates workitem-private memory (registers)
  real apm[MWID];
  real bpm[NWID];
  real cpm[NWID][MWID];

  // Initializes the accumulation registers
  InitAccRegistersDirect(cpm);

  // The faster version of GEMM is not allowed on the (incomplete) borders. Therefore, this section
  // processes only the main parts: output blocks of MWGD by NWGD.
  const int idm = get_local_id(0) * MWID + GetGroupID0() * MWGD;
  const int idn = get_local_id(1) * NWID + GetGroupID1() * NWGD;
  if ((idm < (kSizeM/MWGD)*MWGD) && (idn < (kSizeN/NWGD)*NWGD) &&
      (a_ld % VWMD == 0) && (b_ld % VWND == 0)) {

    // Loops over all complete workgroup tiles
    int kwg = 0;
    for (; kwg < (kSizeK/KWGD) * KWGD; kwg+=KWGD) {

      // Loads data: off-chip --> local (matrix A and B)
      GlobalToLocalDirectA(agm, alm, a_ld, a_offset, tid, kwg, a_transpose, a_conjugate);
      GlobalToLocalDirectB(bgm, blm, b_ld, b_offset, tid, kwg, b_transpose, b_conjugate);
      barrier(CLK_LOCAL_MEM_FENCE);

      // Loops over all workitem tiles, unrolled by a factor KWID
      for (int pwi=0; pwi<KWGD; pwi+=KWID) {
        #pragma unroll
        for (int pit=0; pit<KWID; ++pit) {
          int kg = pwi + pit;

          // Loads data: local --> private (matrix A)
          LocalToPrivateDirectA(alm, apm, kg, a_transpose);

          // Loads data: local --> private (matrix B)
          LocalToPrivateDirectB(blm, bpm, kg, b_transpose);

          // Performs the accumulation (Cpm += Apm * Bpm)
          MultiplyAccumulateDirect(cpm, apm, bpm);
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Loop over the remaining part (incomplete tile in K-dimension)
    for (; kwg < kSizeK; ++kwg) {
      const int idk = kwg;

      // Loads A into register memory
      #pragma unroll
      for (int mi=0; mi<MWID; ++mi) {
        const int a_index = (a_transpose) ? (idm + mi)*a_ld + idk : idk*a_ld + (idm + mi);
        apm[mi] = agms[a_index + a_offset];
        if (a_conjugate) { COMPLEX_CONJUGATE(apm[mi]); }
      }

      // Loads B into register memory
      #pragma unroll
      for (int ni=0; ni<NWID; ++ni) {
        const int b_index = (b_transpose) ? (idn + ni)*b_ld + idk : idk*b_ld + (idn + ni);
        bpm[ni] = bgms[b_index + b_offset];
        if (b_conjugate) { COMPLEX_CONJUGATE(bpm[ni]); }
      }

      // Performs the accumulation (Cpm += Apm * Bpm)
      MultiplyAccumulateDirect(cpm, apm, bpm);
    }

    // Stores a tile of results and performs the multiplication with alpha and beta
    StoreResultsDirect(cgm, cpm, kSizeM, kSizeN, alpha, beta, c_ld, c_offset, c_transpose);
  }

  // Simple but slow version for the parts on the edge (incomplete tiles in M and N-dimensions)
  else {

    // Loop over the K-dimension
    for (int idk = 0; idk < kSizeK; ++idk) {

      // Loads A into register memory
      #pragma unroll
      for (int mi=0; mi<MWID; ++mi) {
        if (idm + mi < kSizeM) {
          const int a_index = (a_transpose) ? (idm + mi)*a_ld + idk : idk*a_ld + (idm + mi);
          apm[mi] = agms[a_index + a_offset];
          if (a_conjugate) { COMPLEX_CONJUGATE(apm[mi]); }
        }
        else {
          SetToZero(apm[mi]);
        }
      }

      // Loads B into register memory
      #pragma unroll
      for (int ni=0; ni<NWID; ++ni) {
        if (idn + ni < kSizeN) {
          const int b_index = (b_transpose) ? (idn + ni)*b_ld + idk : idk*b_ld + (idn + ni);
          bpm[ni] = bgms[b_index + b_offset];
          if (b_conjugate) { COMPLEX_CONJUGATE(bpm[ni]); }
        }
        else {
          SetToZero(bpm[ni]);
        }
      }

      // Performs the accumulation (Cpm += Apm * Bpm)
      MultiplyAccumulateDirect(cpm, apm, bpm);
    }

    // Stores the results
    #pragma unroll
    for (int ni=0; ni<NWID; ++ni) {
      #pragma unroll
      for (int mi=0; mi<MWID; ++mi) {
        if ((idm + mi) < kSizeM && (idn + ni) < kSizeN) {

          // Determines the destination index
          const int c_index = (c_transpose) ? (idm + mi)*c_ld + (idn + ni) : (idn + ni)*c_ld + (idm + mi);

          // Computes and stores the result
          real result;
          AXPBY(result, alpha, cpm[ni][mi], beta, cgm[c_index + c_offset]);
          cgm[c_index + c_offset] = result;
        }
      }
    }
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
