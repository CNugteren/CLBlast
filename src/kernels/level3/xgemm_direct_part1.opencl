
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
// This kernel is seperated into three files. This is part 1 out of 3.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library. Note that all parameters here have a
// suffix 'D' to denote that they are for the 'direct' version of the GEMM kernel.
#ifndef WGD
  #define WGD 8      // Tile-size in dimension M, N, and K (e.g. 8, 16, 32, 64)
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
  #define KWID 1      // Unroll factor of the WGD loop (smaller or equal than WGD)
#endif
#ifndef VWMD
  #define VWMD 1      // Vector width of matrices A and C
#endif
#ifndef VWND
  #define VWND 1      // Vector width of matrix B
#endif
#ifndef PADA
  #define PADA 1      // Local memory padding for matrix A
#endif
#ifndef PADB
  #define PADB 1      // Local memory padding for matrix B
#endif

// Helper parameters based on the above tuning parameters
#define MWID (WGD/MDIMCD)                // Work per work-item (M-dimension)
#define NWID (WGD/NDIMCD)                // Work per work-item (N-dimension)
#define KDIMAD ((MDIMCD*NDIMCD)/(MDIMAD)) // Re-shaped tile dimension of matrix A: KDIMAD * MDIMAD
#define KDIMBD ((MDIMCD*NDIMCD)/(NDIMBD)) // Re-shaped tile dimension of matrix B: KDIMBD * NDIMBD
#define MWAD (WGD/MDIMAD)                // Amount of loads-per-thread for matrix A (M-dimension)
#define KWAD (WGD/KDIMAD)                // Amount of loads-per-thread for matrix A (K-dimension)
#define KWBD (WGD/KDIMBD)                // Amount of loads-per-thread for matrix B (K-dimension)
#define NWBD (WGD/NDIMBD)                // Amount of loads-per-thread for matrix B (N-dimension)

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

// Loads global off-chip memory into thread-private register files. This function is specific for
// loading the A input matrix.
inline void GlobalToPrivateDirectA(const __global real* restrict agms, real apm[MWID],
                                   const int a_ld, const int a_offset, const int idm, const int idk,
                                   const int a_transpose, const int a_conjugate) {
  #pragma unroll
  for (int mi=0; mi<MWID; ++mi) {
    const int a_index = (a_transpose) ? (idm + mi)*a_ld + idk : idk*a_ld + (idm + mi);
    apm[mi] = agms[a_index + a_offset];
    if (a_conjugate) { COMPLEX_CONJUGATE(apm[mi]); }
  }
}

// Same as above, but now for the B input matrix
inline void GlobalToPrivateDirectB(const __global real* restrict bgms, real bpm[NWID],
                                   const int b_ld, const int b_offset, const int idn, const int idk,
                                   const int b_transpose, const int b_conjugate) {
  #pragma unroll
  for (int ni=0; ni<NWID; ++ni) {
    const int b_index = (b_transpose) ? (idn + ni)*b_ld + idk : idk*b_ld + (idn + ni);
    bpm[ni] = bgms[b_index + b_offset];
    if (b_conjugate) { COMPLEX_CONJUGATE(bpm[ni]); }
  }
}

// Loads global off-chip memory into thread-private register files. This function is specific for
// loading the A input matrix. This is the same as above but now includes a bounds check.
inline void GlobalToPrivateCheckedA(const __global real* restrict agms, real apm[MWID],
                                    const int a_ld, const int a_offset, const int idm, const int idk,
                                    const int a_transpose, const int a_conjugate,
                                    const int kSizeM) {
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
}

// Same as above, but now for the B input matrix
inline void GlobalToPrivateCheckedB(const __global real* restrict bgms, real bpm[NWID],
                                    const int b_ld, const int b_offset, const int idn, const int idk,
                                    const int b_transpose, const int b_conjugate,
                                    const int kSizeN) {
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
}

// =================================================================================================

// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication
// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm
inline void StoreResultsDirect(__global real* cgm, real cpm[NWID][MWID],
                               const int idm, const int idn,
                               const real alpha, const real beta,
                               const int c_ld, const int c_offset, const int c_transpose) {
  #pragma unroll
  for (int ni=0; ni<NWID; ++ni) {
    #pragma unroll
    for (int mi=0; mi<MWID; ++mi) {

      // Determines the destination index
      int c_index = (c_transpose) ? (idm + mi)*c_ld + (idn + ni) : (idn + ni)*c_ld + (idm + mi);

      // The final multiplication with alpha (in case beta == 0)
      real result;
      if (IsZero(beta)) {
        Multiply(result, alpha, cpm[ni][mi]);
      }
      // The final multiplication with alpha and the addition with beta*C
      else {
        AXPBY(result, alpha, cpm[ni][mi], beta, cgm[c_index + c_offset]);
      }
      cgm[c_index + c_offset] = result;
    }
  }
}

// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication
// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm
inline void StoreResultsChecked(__global real* cgm, real cpm[NWID][MWID],
                                const int idm, const int idn, const int kSizeM, const int kSizeN,
                                const real alpha, const real beta,
                                const int c_ld, const int c_offset, const int c_transpose) {
  #pragma unroll
  for (int ni=0; ni<NWID; ++ni) {
    #pragma unroll
    for (int mi=0; mi<MWID; ++mi) {
      if ((idm + mi) < kSizeM && (idn + ni) < kSizeN) {

        // Determines the destination index
        int c_index = (c_transpose) ? (idm + mi)*c_ld + (idn + ni) : (idn + ni)*c_ld + (idm + mi);

        // The final multiplication with alpha (in case beta == 0)
        real result;
        if (IsZero(beta)) {
          Multiply(result, alpha, cpm[ni][mi]);
        }
        // The final multiplication with alpha and the addition with beta*C
        else {
          AXPBY(result, alpha, cpm[ni][mi], beta, cgm[c_index + c_offset]);
        }
        cgm[c_index + c_offset] = result;
      }
    }
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
