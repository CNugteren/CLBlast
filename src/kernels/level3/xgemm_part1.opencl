
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains an optimized matrix-multiplication kernel inspired by the paper by Matsumoto
// et al. and the tutorial on http://www.cedricnugteren.nl/tutorial.php. It is fully configurable
// (and tunable!) using more or less the same parameters/naming conventions as in the paper. It
// supports different data-types (SGEMM/DGEMM/CGEMM/ZGEMM/HGEMM) through a pre-processor define.
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
// This kernel is seperated into three files. This is part 1 out of 3.
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
#ifndef USE_VECTOR_MAD
  #define USE_VECTOR_MAD 0      // Unroll (0) or don't (1) unroll the vector MAD manually
#endif
#ifndef GLOBAL_MEM_FENCE
  #define GLOBAL_MEM_FENCE 0    // Global synchronisation barrier for potential better performance
#endif

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
INLINE_FUNC void InitAccRegisters(realM cpm[NWI][MWI/VWM]) {
  #pragma unroll
  for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
    #pragma unroll
    for (int _ni = 0; _ni < NWI; _ni += 1) {
      #if VWM == 1
        SetToZero(cpm[_ni][_mi]);
      #elif VWM == 2
        SetToZero(cpm[_ni][_mi].x);
        SetToZero(cpm[_ni][_mi].y);
      #elif VWM == 4
        SetToZero(cpm[_ni][_mi].x);
        SetToZero(cpm[_ni][_mi].y);
        SetToZero(cpm[_ni][_mi].z);
        SetToZero(cpm[_ni][_mi].w);
      #elif VWM == 8
        SetToZero(cpm[_ni][_mi].s0);
        SetToZero(cpm[_ni][_mi].s1);
        SetToZero(cpm[_ni][_mi].s2);
        SetToZero(cpm[_ni][_mi].s3);
        SetToZero(cpm[_ni][_mi].s4);
        SetToZero(cpm[_ni][_mi].s5);
        SetToZero(cpm[_ni][_mi].s6);
        SetToZero(cpm[_ni][_mi].s7);
      #elif VWM == 16
        SetToZero(cpm[_ni][_mi].s0);
        SetToZero(cpm[_ni][_mi].s1);
        SetToZero(cpm[_ni][_mi].s2);
        SetToZero(cpm[_ni][_mi].s3);
        SetToZero(cpm[_ni][_mi].s4);
        SetToZero(cpm[_ni][_mi].s5);
        SetToZero(cpm[_ni][_mi].s6);
        SetToZero(cpm[_ni][_mi].s7);
        SetToZero(cpm[_ni][_mi].s8);
        SetToZero(cpm[_ni][_mi].s9);
        SetToZero(cpm[_ni][_mi].sA);
        SetToZero(cpm[_ni][_mi].sB);
        SetToZero(cpm[_ni][_mi].sC);
        SetToZero(cpm[_ni][_mi].sD);
        SetToZero(cpm[_ni][_mi].sE);
        SetToZero(cpm[_ni][_mi].sF);
      #endif
    }
  }
}

// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix.
#if SA == 1
INLINE_FUNC void GlobalToLocalA(const __global realM* restrict agm, LOCAL_PTR realM* alm,
                                const int kSizeM, const int tid, const int kwg) {
  const int la0 = tid % MDIMA;
  const int la1 = tid / MDIMA;
  #pragma unroll
  for (int _mia = 0; _mia < MWA/VWM; _mia += 1) {
    #pragma unroll
    for (int _kia = 0; _kia < KWA; _kia += 1) {

      // Computes the indices based on strided/non-strided access
      #if STRM == 0
        int mg = _mia + la0*(MWA/VWM);
      #elif STRM == 1
        int mg = la0 + _mia*MDIMA;
      #endif

      // Computes the indices for the global memory
      int kg = _kia + la1*KWA;
      int idm = mg + GetGroupID0() * (MWG/VWM);
      int idk = kg + kwg;

      // Loads the data from global memory (not transposed) into the local memory
      alm[kg*(MWG/VWM) + mg] = agm[idk*(kSizeM/VWM) + idm];
    }
  }
}
#endif

// Same as above, but now for the B input matrix
#if SB == 1
INLINE_FUNC void GlobalToLocalB(const __global realN* restrict bgm, LOCAL_PTR realN* blm,
                                const int kSizeN, const int tid, const int kwg) {
  const int lb0 = tid % NDIMB;
  const int lb1 = tid / NDIMB;
  #pragma unroll
  for (int _kib = 0; _kib < KWB; _kib += 1) {
    #pragma unroll
    for (int _nib = 0; _nib < NWB/VWN; _nib += 1) {

      // Computes the indices based on strided/non-strided access
      #if STRN == 0
        int ng = _nib + lb0*(NWB/VWN);
      #elif STRN == 1
        int ng = lb0 + _nib*NDIMB;
      #endif

      // Computes the indices for the global memory
      int kg = _kib + lb1*KWB;
      int idn = ng + GetGroupID1() * (NWG/VWN);
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
INLINE_FUNC void GlobalToPrivateA(const __global realM* restrict agm, realM apm[MWI/VWM],
                                  const int kSizeM, const int idk, const int kwg) {
  #pragma unroll
  for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {

    // Computes the indices based on strided/non-strided access
    #if STRM == 0
      int mg = _mi + get_local_id(0)*(MWI/VWM);
    #elif STRM == 1
      int mg = get_local_id(0) + _mi*MDIMC;
    #endif

    // Computes the indices for the global memory
    int idm = mg + GetGroupID0() * (MWG/VWM);

    // Loads the data from global memory (not transposed) and stores into registers
    apm[_mi] = agm[idk*(kSizeM/VWM) + idm];
  }
}
#endif

// Same as above, but now for the B input matrix
#if SB == 0
INLINE_FUNC void GlobalToPrivateB(const __global realN* restrict bgm, realN bpm[NWI/VWN],
                                  const int kSizeN, const int idk) {
  #pragma unroll
  for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {

    // Computes the indices based on strided/non-strided access
    #if STRN == 0
      int ng = _ni + get_local_id(1)*(NWI/VWN);
    #elif STRN == 1
      int ng = get_local_id(1) + _ni*NDIMC;
    #endif

    // Computes the indices for the global memory
    int idn = ng + GetGroupID1() * (NWG/VWN);

    // Loads the data from global memory (transposed) and stores into registers
    bpm[_ni] = bgm[idk*(kSizeN/VWN) + idn];
  }
}
#endif

// =================================================================================================

// Caches on-chip local memory into per-thread private memory (registers). This function is specific
// for caching the A input matrix.
#if SA == 1
INLINE_FUNC void LocalToPrivateA(LOCAL_PTR realM* alm, realM apm[MWI/VWM], const int kg) {
  #pragma unroll
  for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
    #if STRM == 0
      int mg = _mi + get_local_id(0)*(MWI/VWM);
    #elif STRM == 1
      int mg = get_local_id(0) + _mi*MDIMC;
    #endif
    apm[_mi] = alm[kg*(MWG/VWM) + mg];
  }
}
#endif

// Same as above, but now for the B input matrix
#if SB == 1
INLINE_FUNC void LocalToPrivateB(LOCAL_PTR realN* blm, realN bpm[NWI/VWN], const int kg) {
  #pragma unroll
  for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
    #if STRN == 0
      int ng = _ni + get_local_id(1)*(NWI/VWN);
    #elif STRN == 1
      int ng = get_local_id(1) + _ni*NDIMC;
    #endif
    bpm[_ni] = blm[kg*(NWG/VWN) + ng];
  }
}
#endif

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
