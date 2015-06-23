
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains an optimized matrix-multiplication kernel according to the paper by Matsumoto
// et al. and the tutorial on http://www.cedricnugteren.nl/tutorial.php. It is fully configurable
// (and tunable!) using more or less the same parameters/naming conventions as in the paper. It
// supports single and double precision (SGEMM/DGEMM) through a pre-processor define.
//
// Matrices are accessed as follows:
// A: [k*M + m], with 'k' ranging from 0:K and 'm' from 0:M (m,k,m)
// B: [k*N + n], with 'k' ranging from 0:K and 'n' from 0:N (n,k,n)
// C: [n*M + m], with 'n' ranging from 0:N and 'm' from 0:M (m,n,m)
//
// Or as an image (assuming column-major)
//       K                      
//    o-------o                 
//    |       |                 
//  N | [B^T] |                 
//    |       |                 
//    o-------o                 
//        K               N     
//    o-------o        o-----o  
//  M |  [A]  |      M | [C] |  
//    |       |        |     |  
//    o-------o        o-----o  
//                              
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.
#ifndef MWG
  #define MWG 8      // Tile-size in dimension M (e.g. 64, 128)
#endif
#ifndef NWG
  #define NWG 8      // Tile-size in dimension N (e.g. 64, 128)
#endif
#ifndef KWG
  #define KWG 8      // Tile-size in dimension K (e.g. 8, 16)
#endif
#ifndef MDIMC
  #define MDIMC 8    // Threads per workgroup in M-dimension (e.g. 8, 16, 32)
#endif
#ifndef NDIMC
  #define NDIMC 8    // Threads per workgroup in N-dimension (e.g. 8, 16, 32)
#endif
#ifndef MDIMA
  #define MDIMA 8    // Re-shaped tile dimension of matrix A: KDIMA * MDIMA
#endif
#ifndef NDIMB
  #define NDIMB 8    // Re-shaped tile dimension of matrix B: KDIMB * NDIMB
#endif
#ifndef KWI
  #define KWI 1      // Unroll factor of the KWG loop (smaller or equal than KWG)
#endif
#ifndef VWM
  #define VWM 1      // Vector width of matrices A and C 
#endif
#ifndef VWN
  #define VWN 1      // Vector width of matrix B
#endif
#ifndef STRM
  #define STRM 0     // Use strided access within a thread in the M-dimension (1) or not (0)
#endif
#ifndef STRN
  #define STRN 0     // Use strided access within a thread in the N-dimension (1) or not (0)
#endif
#ifndef SA
  #define SA 0       // Use local/shared memory to cache matrix A (1) or not (0)
#endif
#ifndef SB
  #define SB 0       // Use local/shared memory to cache matrix B (1) or not (0)
#endif

// Helper parameters based on the above tuning parameters
#define MWI (MWG/MDIMC)               // Work per work-item (M-dimension)
#define NWI (NWG/NDIMC)               // Work per work-item (N-dimension)
#define KDIMA ((MDIMC*NDIMC)/(MDIMA)) // Re-shaped tile dimension of matrix A: KDIMA * MDIMA
#define KDIMB ((MDIMC*NDIMC)/(NDIMB)) // Re-shaped tile dimension of matrix B: KDIMB * NDIMB
#define MWA (MWG/MDIMA)               // Amount of loads-per-thread for matrix A (M-dimension)
#define KWA (KWG/KDIMA)               // Amount of loads-per-thread for matrix A (K-dimension)
#define KWB (KWG/KDIMB)               // Amount of loads-per-thread for matrix B (K-dimension)
#define NWB (NWG/NDIMB)               // Amount of loads-per-thread for matrix B (N-dimension)

// Settings
#define USE_VECTOR_MAD 0              // Unroll (0) or don't (1) unroll the vector MAD manually

// =================================================================================================

// Data-widths in dimension M
#if VWM == 1
    typedef real realM;
#elif VWM == 2
    typedef real2 realM;
#elif VWM == 4
    typedef real4 realM;
#elif VWM == 8
    typedef real8 realM;
#elif VWM == 16
    typedef real16 realM;
#endif

// Data-widths in dimension N
#if VWN == 1
    typedef real realN;
#elif VWN == 2
    typedef real2 realN;
#elif VWN == 4
    typedef real4 realN;
#elif VWN == 8
    typedef real8 realN;
#elif VWN == 16
    typedef real16 realN;
#endif

// =================================================================================================

// Initializes the accumulation registers to zero
inline void InitAccRegisters(realM cpm[NWI][MWI/VWM]) {
  #pragma unroll
  for (int mi=0; mi<MWI/VWM; ++mi) {
    #pragma unroll
    for (int ni=0; ni<NWI; ++ni) {
      #if VWM == 1
        SetToZero(cpm[ni][mi]);
      #elif VWM == 2
        SetToZero(cpm[ni][mi].x);
        SetToZero(cpm[ni][mi].y);
      #elif VWM == 4
        SetToZero(cpm[ni][mi].x);
        SetToZero(cpm[ni][mi].y);
        SetToZero(cpm[ni][mi].z);
        SetToZero(cpm[ni][mi].w);
      #elif VWM == 8
        SetToZero(cpm[ni][mi].s0);
        SetToZero(cpm[ni][mi].s1);
        SetToZero(cpm[ni][mi].s2);
        SetToZero(cpm[ni][mi].s3);
        SetToZero(cpm[ni][mi].s4);
        SetToZero(cpm[ni][mi].s5);
        SetToZero(cpm[ni][mi].s6);
        SetToZero(cpm[ni][mi].s7);
      #elif VWM == 16
        SetToZero(cpm[ni][mi].s0);
        SetToZero(cpm[ni][mi].s1);
        SetToZero(cpm[ni][mi].s2);
        SetToZero(cpm[ni][mi].s3);
        SetToZero(cpm[ni][mi].s4);
        SetToZero(cpm[ni][mi].s5);
        SetToZero(cpm[ni][mi].s6);
        SetToZero(cpm[ni][mi].s7);
        SetToZero(cpm[ni][mi].s8);
        SetToZero(cpm[ni][mi].s9);
        SetToZero(cpm[ni][mi].sA);
        SetToZero(cpm[ni][mi].sB);
        SetToZero(cpm[ni][mi].sC);
        SetToZero(cpm[ni][mi].sD);
        SetToZero(cpm[ni][mi].sE);
        SetToZero(cpm[ni][mi].sF);
      #endif
    }
  }
}

// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix.
#if SA == 1
inline void GlobalToLocalA(const __global realM* restrict agm, __local realM* alm,
                           const int kSizeM, const int tid, const int kwg) {
  const int la0 = tid % MDIMA;
  const int la1 = tid / MDIMA;
  #pragma unroll
  for (int mia=0; mia<MWA/VWM; ++mia) {
    #pragma unroll
    for (int kia=0; kia<KWA; ++kia) {

      // Computes the indices based on strided/non-strided access
      #if STRM == 0
        int mg = mia + la0*(MWA/VWM);
      #elif STRM == 1
        int mg = la0 + mia*MDIMA;
      #endif

      // Computes the indices for the global memory
      int kg = kia + la1*KWA;
      int idm = mg + get_group_id(0)*(MWG/VWM);
      int idk = kg + kwg;

      // Loads the data from global memory (not transposed) into the local memory
      alm[kg*(MWG/VWM) + mg] = agm[idk*(kSizeM/VWM) + idm];
    }
  }
}
#endif

// Same as above, but now for the B input matrix
#if SB == 1
inline void GlobalToLocalB(const __global realN* restrict bgm, __local realN* blm,
                           const int kSizeN, const int tid, const int kwg) {
  const int lb0 = tid % NDIMB;
  const int lb1 = tid / NDIMB;
  #pragma unroll
  for (int kib=0; kib<KWB; ++kib) {
    #pragma unroll
    for (int nib=0; nib<NWB/VWN; ++nib) {

      // Computes the indices based on strided/non-strided access
      #if STRN == 0
        int ng = nib + lb0*(NWB/VWN);
      #elif STRN == 1
        int ng = lb0 + nib*NDIMB;
      #endif

      // Computes the indices for the global memory
      int kg = kib + lb1*KWB;
      int idn = ng + get_group_id(1)*(NWG/VWN);
      int idk = kg + kwg;

      // Loads the data from global memory (transposed) into the local memory
      blm[kg*(NWG/VWN) + ng] = bgm[idk*(kSizeN/VWN) + idn];
    }
  }
}
#endif

// =================================================================================================

// Caches global off-chip memory directly into per-thread private memory (registers). This function
// is specific for caching the A input matrix.
#if SA == 0
inline void GlobalToPrivateA(const __global realM* restrict agm, realM apm[MWI/VWM],
                             const int kSizeM, const int idk, const int kwg) {
  #pragma unroll
  for (int mi=0; mi<MWI/VWM; ++mi) {

    // Computes the indices based on strided/non-strided access
    #if STRM == 0
      int mg = mi + get_local_id(0)*(MWI/VWM);
    #elif STRM == 1
      int mg = get_local_id(0) + mi*MDIMC;
    #endif

    // Computes the indices for the global memory
    int idm = mg + get_group_id(0)*(MWG/VWM);

    // Loads the data from global memory (not transposed) and stores into registers
    apm[mi] = agm[idk*(kSizeM/VWM) + idm];
  }
}
#endif

// Same as above, but now for the B input matrix
#if SB == 0
inline void GlobalToPrivateB(const __global realN* restrict bgm, realN bpm[NWI/VWN],
                             const int kSizeN, const int idk) {
  #pragma unroll
  for (int ni=0; ni<NWI/VWN; ++ni) {

    // Computes the indices based on strided/non-strided access
    #if STRN == 0
      int ng = ni + get_local_id(1)*(NWI/VWN);
    #elif STRN == 1
      int ng = get_local_id(1) + ni*NDIMC;
    #endif

    // Computes the indices for the global memory
    int idn = ng + get_group_id(1)*(NWG/VWN);

    // Loads the data from global memory (transposed) and stores into registers
    bpm[ni] = bgm[idk*(kSizeN/VWN) + idn];
  }
}
#endif

// =================================================================================================

// Caches on-chip local memory into per-thread private memory (registers). This function is specific
// for caching the A input matrix.
#if SA == 1
inline void LocalToPrivateA(__local realM* alm, realM apm[MWI/VWM], const int kg) {
  #pragma unroll
  for (int mi=0; mi<MWI/VWM; ++mi) {
    #if STRM == 0
      int mg = mi + get_local_id(0)*(MWI/VWM);
    #elif STRM == 1
      int mg = get_local_id(0) + mi*MDIMC;
    #endif
    apm[mi] = alm[kg*(MWG/VWM) + mg];
  }
}
#endif

// Same as above, but now for the B input matrix
#if SB == 1
inline void LocalToPrivateB(__local realN* blm, realN bpm[NWI/VWN], const int kg) {
  #pragma unroll
  for (int ni=0; ni<NWI/VWN; ++ni) {
    #if STRN == 0
      int ng = ni + get_local_id(1)*(NWI/VWN);
    #elif STRN == 1
      int ng = get_local_id(1) + ni*NDIMC;
    #endif
    bpm[ni] = blm[kg*(NWG/VWN) + ng];
  }
}
#endif

// =================================================================================================

// The vectorised multiply-add function
inline realM MultiplyAddVector(realM cvec, const realM avec, const real bval) {
  #if USE_VECTOR_MAD == 1
    cvec += avec * bval;
  #else
    #if VWM == 1
      MultiplyAdd(cvec,    avec,    bval);
    #elif VWM == 2
      MultiplyAdd(cvec.x , avec.x,  bval);
      MultiplyAdd(cvec.y , avec.y,  bval);
    #elif VWM == 4
      MultiplyAdd(cvec.x , avec.x,  bval);
      MultiplyAdd(cvec.y , avec.y,  bval);
      MultiplyAdd(cvec.z , avec.z,  bval);
      MultiplyAdd(cvec.w , avec.w,  bval);
    #elif VWM == 8
      MultiplyAdd(cvec.s0, avec.s0, bval);
      MultiplyAdd(cvec.s1, avec.s1, bval);
      MultiplyAdd(cvec.s2, avec.s2, bval);
      MultiplyAdd(cvec.s3, avec.s3, bval);
      MultiplyAdd(cvec.s4, avec.s4, bval);
      MultiplyAdd(cvec.s5, avec.s5, bval);
      MultiplyAdd(cvec.s6, avec.s6, bval);
      MultiplyAdd(cvec.s7, avec.s7, bval);
    #elif VWM == 16
      MultiplyAdd(cvec.s0, avec.s0, bval);
      MultiplyAdd(cvec.s1, avec.s1, bval);
      MultiplyAdd(cvec.s2, avec.s2, bval);
      MultiplyAdd(cvec.s3, avec.s3, bval);
      MultiplyAdd(cvec.s4, avec.s4, bval);
      MultiplyAdd(cvec.s5, avec.s5, bval);
      MultiplyAdd(cvec.s6, avec.s6, bval);
      MultiplyAdd(cvec.s7, avec.s7, bval);
      MultiplyAdd(cvec.s8, avec.s8, bval);
      MultiplyAdd(cvec.s9, avec.s9, bval);
      MultiplyAdd(cvec.sA, avec.sA, bval);
      MultiplyAdd(cvec.sB, avec.sB, bval);
      MultiplyAdd(cvec.sC, avec.sC, bval);
      MultiplyAdd(cvec.sD, avec.sD, bval);
      MultiplyAdd(cvec.sE, avec.sE, bval);
      MultiplyAdd(cvec.sF, avec.sF, bval);
    #endif
  #endif
  return cvec;
}

// Performs the actual computation: Cpm += Apm * Bpm
inline void MultiplyAccumulate(realM cpm[NWI][MWI/VWM], realM apm[MWI/VWM], realN bpm[NWI/VWN]) {
  #pragma unroll
  for (int ni=0; ni<NWI/VWN; ++ni) {
    #pragma unroll
    for (int mi=0; mi<MWI/VWM; ++mi) {
      #if VWN == 1
        cpm[ni*VWN + 0][mi] = MultiplyAddVector(cpm[ni*VWN + 0][mi], apm[mi], bpm[ni]);
      #elif VWN == 2
        cpm[ni*VWN + 0][mi] = MultiplyAddVector(cpm[ni*VWN + 0][mi], apm[mi], bpm[ni].x);
        cpm[ni*VWN + 1][mi] = MultiplyAddVector(cpm[ni*VWN + 1][mi], apm[mi], bpm[ni].y);
      #elif VWN == 4
        cpm[ni*VWN + 0][mi] = MultiplyAddVector(cpm[ni*VWN + 0][mi], apm[mi], bpm[ni].x);
        cpm[ni*VWN + 1][mi] = MultiplyAddVector(cpm[ni*VWN + 1][mi], apm[mi], bpm[ni].y);
        cpm[ni*VWN + 2][mi] = MultiplyAddVector(cpm[ni*VWN + 2][mi], apm[mi], bpm[ni].z);
        cpm[ni*VWN + 3][mi] = MultiplyAddVector(cpm[ni*VWN + 3][mi], apm[mi], bpm[ni].w);
      #elif VWN == 8
        cpm[ni*VWN + 0][mi] = MultiplyAddVector(cpm[ni*VWN + 0][mi], apm[mi], bpm[ni].s0);
        cpm[ni*VWN + 1][mi] = MultiplyAddVector(cpm[ni*VWN + 1][mi], apm[mi], bpm[ni].s1);
        cpm[ni*VWN + 2][mi] = MultiplyAddVector(cpm[ni*VWN + 2][mi], apm[mi], bpm[ni].s2);
        cpm[ni*VWN + 3][mi] = MultiplyAddVector(cpm[ni*VWN + 3][mi], apm[mi], bpm[ni].s3);
        cpm[ni*VWN + 4][mi] = MultiplyAddVector(cpm[ni*VWN + 4][mi], apm[mi], bpm[ni].s4);
        cpm[ni*VWN + 5][mi] = MultiplyAddVector(cpm[ni*VWN + 5][mi], apm[mi], bpm[ni].s5);
        cpm[ni*VWN + 6][mi] = MultiplyAddVector(cpm[ni*VWN + 6][mi], apm[mi], bpm[ni].s6);
        cpm[ni*VWN + 7][mi] = MultiplyAddVector(cpm[ni*VWN + 7][mi], apm[mi], bpm[ni].s7);
      #elif VWN == 16
        cpm[ni*VWN + 0 ][mi] = MultiplyAddVector(cpm[ni*VWN + 0 ][mi], apm[mi], bpm[ni].s0);
        cpm[ni*VWN + 1 ][mi] = MultiplyAddVector(cpm[ni*VWN + 1 ][mi], apm[mi], bpm[ni].s1);
        cpm[ni*VWN + 2 ][mi] = MultiplyAddVector(cpm[ni*VWN + 2 ][mi], apm[mi], bpm[ni].s2);
        cpm[ni*VWN + 3 ][mi] = MultiplyAddVector(cpm[ni*VWN + 3 ][mi], apm[mi], bpm[ni].s3);
        cpm[ni*VWN + 4 ][mi] = MultiplyAddVector(cpm[ni*VWN + 4 ][mi], apm[mi], bpm[ni].s4);
        cpm[ni*VWN + 5 ][mi] = MultiplyAddVector(cpm[ni*VWN + 5 ][mi], apm[mi], bpm[ni].s5);
        cpm[ni*VWN + 6 ][mi] = MultiplyAddVector(cpm[ni*VWN + 6 ][mi], apm[mi], bpm[ni].s6);
        cpm[ni*VWN + 7 ][mi] = MultiplyAddVector(cpm[ni*VWN + 7 ][mi], apm[mi], bpm[ni].s7);
        cpm[ni*VWN + 8 ][mi] = MultiplyAddVector(cpm[ni*VWN + 8 ][mi], apm[mi], bpm[ni].s8);
        cpm[ni*VWN + 9 ][mi] = MultiplyAddVector(cpm[ni*VWN + 9 ][mi], apm[mi], bpm[ni].s9);
        cpm[ni*VWN + 10][mi] = MultiplyAddVector(cpm[ni*VWN + 10][mi], apm[mi], bpm[ni].sA);
        cpm[ni*VWN + 11][mi] = MultiplyAddVector(cpm[ni*VWN + 11][mi], apm[mi], bpm[ni].sB);
        cpm[ni*VWN + 12][mi] = MultiplyAddVector(cpm[ni*VWN + 12][mi], apm[mi], bpm[ni].sC);
        cpm[ni*VWN + 13][mi] = MultiplyAddVector(cpm[ni*VWN + 13][mi], apm[mi], bpm[ni].sD);
        cpm[ni*VWN + 14][mi] = MultiplyAddVector(cpm[ni*VWN + 14][mi], apm[mi], bpm[ni].sE);
        cpm[ni*VWN + 15][mi] = MultiplyAddVector(cpm[ni*VWN + 15][mi], apm[mi], bpm[ni].sF);
      #endif
    }
  }
}

// =================================================================================================

// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication
// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm
inline void StoreResults(__global realM* cgm, realM cpm[NWI][MWI/VWM], const int kSizeM,
                         const real alpha, const real beta) {
  #pragma unroll
  for (int ni=0; ni<NWI; ++ni) {
    #pragma unroll
    for (int mi=0; mi<MWI/VWM; ++mi) {
      #if STRM == 0
        int mg = mi + get_local_id(0)*(MWI/VWM);
      #elif STRM == 1
        int mg = get_local_id(0) + mi*MDIMC;
      #endif
      #if STRN == 0
        int ng = ni + get_local_id(1)*NWI;
      #elif STRN == 1
        int ng = ni%VWN + get_local_id(1)*VWN + (ni/VWN)*VWN*NDIMC;
      #endif
      int idm = mg + get_group_id(0)*(MWG/VWM);
      int idn = ng + get_group_id(1)*NWG;

      // The final multiplication with alpha and the addition with beta*C
      int index = idn*(kSizeM/VWM) + idm;
      realM cval = cgm[index];
      #if VWM == 1
        AXPBY(cgm[index], alpha, cpm[ni][mi], beta, cval);
      #elif VWM == 2
        AXPBY(cgm[index].x, alpha, cpm[ni][mi].x, beta, cval.x);
        AXPBY(cgm[index].y, alpha, cpm[ni][mi].y, beta, cval.y);
      #elif VWM == 4
        AXPBY(cgm[index].x, alpha, cpm[ni][mi].x, beta, cval.x);
        AXPBY(cgm[index].y, alpha, cpm[ni][mi].y, beta, cval.y);
        AXPBY(cgm[index].z, alpha, cpm[ni][mi].z, beta, cval.z);
        AXPBY(cgm[index].w, alpha, cpm[ni][mi].w, beta, cval.w);
      #elif VWM == 8
        AXPBY(cgm[index].s0, alpha, cpm[ni][mi].s0, beta, cval.s0);
        AXPBY(cgm[index].s1, alpha, cpm[ni][mi].s1, beta, cval.s1);
        AXPBY(cgm[index].s2, alpha, cpm[ni][mi].s2, beta, cval.s2);
        AXPBY(cgm[index].s3, alpha, cpm[ni][mi].s3, beta, cval.s3);
        AXPBY(cgm[index].s4, alpha, cpm[ni][mi].s4, beta, cval.s4);
        AXPBY(cgm[index].s5, alpha, cpm[ni][mi].s5, beta, cval.s5);
        AXPBY(cgm[index].s6, alpha, cpm[ni][mi].s6, beta, cval.s6);
        AXPBY(cgm[index].s7, alpha, cpm[ni][mi].s7, beta, cval.s7);
      #elif VWM == 16
        AXPBY(cgm[index].s0, alpha, cpm[ni][mi].s0, beta, cval.s0);
        AXPBY(cgm[index].s1, alpha, cpm[ni][mi].s1, beta, cval.s1);
        AXPBY(cgm[index].s2, alpha, cpm[ni][mi].s2, beta, cval.s2);
        AXPBY(cgm[index].s3, alpha, cpm[ni][mi].s3, beta, cval.s3);
        AXPBY(cgm[index].s4, alpha, cpm[ni][mi].s4, beta, cval.s4);
        AXPBY(cgm[index].s5, alpha, cpm[ni][mi].s5, beta, cval.s5);
        AXPBY(cgm[index].s6, alpha, cpm[ni][mi].s6, beta, cval.s6);
        AXPBY(cgm[index].s7, alpha, cpm[ni][mi].s7, beta, cval.s7);
        AXPBY(cgm[index].s8, alpha, cpm[ni][mi].s8, beta, cval.s8);
        AXPBY(cgm[index].s9, alpha, cpm[ni][mi].s9, beta, cval.s9);
        AXPBY(cgm[index].sA, alpha, cpm[ni][mi].sA, beta, cval.sA);
        AXPBY(cgm[index].sB, alpha, cpm[ni][mi].sB, beta, cval.sB);
        AXPBY(cgm[index].sC, alpha, cpm[ni][mi].sC, beta, cval.sC);
        AXPBY(cgm[index].sD, alpha, cpm[ni][mi].sD, beta, cval.sD);
        AXPBY(cgm[index].sE, alpha, cpm[ni][mi].sE, beta, cval.sE);
        AXPBY(cgm[index].sF, alpha, cpm[ni][mi].sF, beta, cval.sF);
      #endif
    }
  }
}

// =================================================================================================

// Main body of the matrix-multiplication algorithm. It calls the (inlined) functions above.
inline void XgemmBody(const int kSizeM, const int kSizeN, const int kSizeK,
                      const __global realM* restrict agm, const __global realN* restrict bgm,
                      __global realM* cgm, realM cpm[NWI][MWI/VWM],
                      #if SA == 1 && SB == 1
                        __local realM* alm, __local realN* blm
                      #elif SA == 1
                        __local realM* alm
                      #elif SB == 1
                        __local realN* blm
                      #endif
                      ) {

  // Allocates workitem-private memory (registers)
  realM apm[MWI/VWM];
  realN bpm[NWI/VWN];

  // Combined thread identifier (volatile to disable caching)
  #if SA == 1 || SB == 1
    volatile int tid = get_local_id(0) + MDIMC*get_local_id(1);
  #endif

  // Initializes the accumulation registers
  InitAccRegisters(cpm);

  // Loops over all workgroup tiles
  for (int kwg=0; kwg<kSizeK; kwg+=KWG) {

    // Loads data: off-chip --> local (matrix A)
    #if SA == 1
      GlobalToLocalA(agm, alm, kSizeM, tid, kwg);
    #endif
    // Loads data: off-chip --> local (matrix B)
    #if SB == 1
      GlobalToLocalB(bgm, blm, kSizeN, tid, kwg);
    #endif
    #if SA == 1 || SB == 1
      barrier(CLK_LOCAL_MEM_FENCE);
    #endif

    // Loops over all workitem tiles, unrolled by a factor KWI
    for (int pwi=0; pwi<KWG; pwi+=KWI) {
      #pragma unroll
      for (int pit=0; pit<KWI; ++pit) {
        #if SA == 0 || SB == 0
          int idk = kwg + pwi + pit;
        #endif
        #if SA == 1 || SB == 1
          int kg = pwi+pit;
        #endif

        // Loads data: local --> private (matrix A)
        #if SA == 1
          LocalToPrivateA(alm, apm, kg);
        // Loads data: off-chip --> private (matrix A)
        #else
          GlobalToPrivateA(agm, apm, kSizeM, idk, kwg);
        #endif

        // Loads data: local --> private (matrix B)
        #if SB == 1
          LocalToPrivateB(blm, bpm, kg);
        // Loads data: off-chip --> private (matrix B)
        #else
          GlobalToPrivateB(bgm, bpm, kSizeN, idk);
        #endif

        // Performs the accumulation (Cpm += Apm * Bpm)
        MultiplyAccumulate(cpm, apm, bpm);
      }
    }
    #if SA == 1 || SB == 1
      barrier(CLK_LOCAL_MEM_FENCE);
    #endif
  }
}

// =================================================================================================

// Main entry point of the kernel. This is the regular full version.
__attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))
__kernel void Xgemm(const int kSizeM, const int kSizeN, const int kSizeK,
                    const real alpha, const real beta,
                    const __global realM* restrict agm,
                    const __global realN* restrict bgm,
                    __global realM* cgm) {

  // Allocates workgroup-private memory (local memory)
  #if SA == 1
    __local realM alm[KWG * MWG/VWM];
  #endif
  #if SB == 1
    __local realN blm[KWG * NWG/VWN];
  #endif

  // Computes the matrix-multiplication and stores the result in register memory
  realM cpm[NWI][MWI/VWM];
  #if SA == 1 && SB == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, cpm, alm, blm);
  #elif SA == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, cpm, alm);
  #elif SB == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, cpm, blm);
  #else
    XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, cpm);
  #endif

  // Stores an MWG * NWG tile of results and performs the multiplication with alpha and beta
  StoreResults(cgm, cpm, kSizeM, alpha, beta);
}

// =================================================================================================

// Main entry point of the kernel. This is the upper-triangular version.
__attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))
__kernel void XgemmUpper(const int kSizeN, const int kSizeK,
                         const real alpha, const real beta,
                         const __global realM* restrict agm,
                         const __global realN* restrict bgm,
                         __global realM* cgm) {

  // Skip these threads if they do not contain threads contributing to the upper-triangle
  if (get_group_id(1)*NWG < get_group_id(0)*MWG) {
    return;
  }

  // Allocates workgroup-private memory (local memory)
  #if SA == 1
    __local realM alm[KWG * MWG/VWM];
  #endif
  #if SB == 1
    __local realN blm[KWG * NWG/VWN];
  #endif

  // Computes the matrix-multiplication and stores the result in register memory
  realM cpm[NWI][MWI/VWM];
  #if SA == 1 && SB == 1
    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, cpm, alm, blm);
  #elif SA == 1
    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, cpm, alm);
  #elif SB == 1
    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, cpm, blm);
  #else
    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, cpm);
  #endif

  // Stores an MWG * NWG tile of results and performs the multiplication with alpha and beta
  StoreResults(cgm, cpm, kSizeN, alpha, beta);
}

// =================================================================================================

// Main entry point of the kernel. This is the lower-triangular version.
__attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))
__kernel void XgemmLower(const int kSizeN, const int kSizeK,
                         const real alpha, const real beta,
                         const __global realM* restrict agm,
                         const __global realN* restrict bgm,
                         __global realM* cgm) {

  // Skip these threads if they do not contain threads contributing to the lower-triangle
  if (get_group_id(1)*NWG > get_group_id(0)*MWG) {
    return;
  }

  // Allocates workgroup-private memory (local memory)
  #if SA == 1
    __local realM alm[KWG * MWG/VWM];
  #endif
  #if SB == 1
    __local realN blm[KWG * NWG/VWN];
  #endif

  // Computes the matrix-multiplication and stores the result in register memory
  realM cpm[NWI][MWI/VWM];
  #if SA == 1 && SB == 1
    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, cpm, alm, blm);
  #elif SA == 1
    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, cpm, alm);
  #elif SB == 1
    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, cpm, blm);
  #else
    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, cpm);
  #endif

  // Stores an MWG * NWG tile of results and performs the multiplication with alpha and beta
  StoreResults(cgm, cpm, kSizeN, alpha, beta);
}

// =================================================================================================

// End of the C++11 raw string literal
)";

// =================================================================================================
