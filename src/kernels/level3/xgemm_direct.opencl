
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

// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix.
inline void GlobalToLocalDirectA(const __global realM* restrict agm, __local real* alm,
                                 const int a_ld, const int a_offset, const int tid, const int kwg,
                                 const int a_transpose, const int a_conjugate) {
  const int la0 = tid % MDIMA;
  const int la1 = tid / MDIMA;
  #pragma unroll
  for (int mia=0; mia<MWA/VWM; ++mia) {
    #pragma unroll
    for (int kia=0; kia<KWA; ++kia) {

      // Computes the indices for the global memory
      int mg = mia + la0*(MWA/VWM);
      int kg = kia + la1*KWA;
      int idm = (a_transpose) ? mg + kwg/VWM : mg + GetGroupID0()*(MWG/VWM);
      int idk = (a_transpose) ? kg + GetGroupID0()*MWG : kg + kwg;

      // Loads the data from global memory into the local memory
      const realM avec = agm[idk*(a_ld/VWM) + idm + a_offset];
      #if VWM == 1
         alm[kg*MWG + mg] = avec;
      #elif VWM == 2
         alm[kg*MWG + mg*VWM + 0] = avec.x;
         alm[kg*MWG + mg*VWM + 1] = avec.y;
      #elif VWM == 4
         alm[kg*MWG + mg*VWM + 0] = avec.x;
         alm[kg*MWG + mg*VWM + 1] = avec.y;
         alm[kg*MWG + mg*VWM + 2] = avec.z;
         alm[kg*MWG + mg*VWM + 3] = avec.w;
      #elif VWM == 8
         alm[kg*MWG + mg*VWM + 0] = avec.s0;
         alm[kg*MWG + mg*VWM + 1] = avec.s1;
         alm[kg*MWG + mg*VWM + 2] = avec.s2;
         alm[kg*MWG + mg*VWM + 3] = avec.s3;
         alm[kg*MWG + mg*VWM + 4] = avec.s4;
         alm[kg*MWG + mg*VWM + 5] = avec.s5;
         alm[kg*MWG + mg*VWM + 6] = avec.s6;
         alm[kg*MWG + mg*VWM + 7] = avec.s7;
      #elif VWM == 16
         alm[kg*MWG + mg*VWM + 0] = avec.s0;
         alm[kg*MWG + mg*VWM + 1] = avec.s1;
         alm[kg*MWG + mg*VWM + 2] = avec.s2;
         alm[kg*MWG + mg*VWM + 3] = avec.s3;
         alm[kg*MWG + mg*VWM + 4] = avec.s4;
         alm[kg*MWG + mg*VWM + 5] = avec.s5;
         alm[kg*MWG + mg*VWM + 6] = avec.s6;
         alm[kg*MWG + mg*VWM + 7] = avec.s7;
         alm[kg*MWG + mg*VWM + 8] = avec.s8;
         alm[kg*MWG + mg*VWM + 9] = avec.s9;
         alm[kg*MWG + mg*VWM + 10] = avec.sA;
         alm[kg*MWG + mg*VWM + 11] = avec.sB;
         alm[kg*MWG + mg*VWM + 12] = avec.sC;
         alm[kg*MWG + mg*VWM + 13] = avec.sD;
         alm[kg*MWG + mg*VWM + 14] = avec.sE;
         alm[kg*MWG + mg*VWM + 15] = avec.sF;
      #endif
      if (a_conjugate) {
        for (int vm=0; vm<VWM; ++vm) {
          COMPLEX_CONJUGATE(alm[kg*MWG + mg*VWM + vm]);
        }
      }
    }
  }
}

// Same as above, but now for the B input matrix
inline void GlobalToLocalDirectB(const __global realN* restrict bgm, __local real* blm,
                                 const int b_ld, const int b_offset, const int tid, const int kwg,
                                 const int b_transpose, const int b_conjugate) {
  const int lb0 = tid % NDIMB;
  const int lb1 = tid / NDIMB;
  #pragma unroll
  for (int kib=0; kib<KWB; ++kib) {
    #pragma unroll
    for (int nib=0; nib<NWB/VWN; ++nib) {

      // Computes the indices for the global memory
      int ng = nib + lb0*(NWB/VWN);
      int kg = kib + lb1*KWB;
      int idn = (b_transpose) ? ng + kwg/VWN : ng + GetGroupID1()*(NWG/VWN);
      int idk = (b_transpose) ? kg + GetGroupID1()*NWG : kg + kwg;

      // Loads the data from global memory into the local memory
      const realM bvec = bgm[idk*(b_ld/VWN) + idn + b_offset];
      #if VWN == 1
         blm[kg*NWG + ng] = bvec;
      #elif VWN == 2
         blm[kg*NWG + ng*VWN + 0] = bvec.x;
         blm[kg*NWG + ng*VWN + 1] = bvec.y;
      #elif VWN == 4
         blm[kg*NWG + ng*VWN + 0] = bvec.x;
         blm[kg*NWG + ng*VWN + 1] = bvec.y;
         blm[kg*NWG + ng*VWN + 2] = bvec.z;
         blm[kg*NWG + ng*VWN + 3] = bvec.w;
      #elif VWN == 8
         blm[kg*NWG + ng*VWN + 0] = bvec.s0;
         blm[kg*NWG + ng*VWN + 1] = bvec.s1;
         blm[kg*NWG + ng*VWN + 2] = bvec.s2;
         blm[kg*NWG + ng*VWN + 3] = bvec.s3;
         blm[kg*NWG + ng*VWN + 4] = bvec.s4;
         blm[kg*NWG + ng*VWN + 5] = bvec.s5;
         blm[kg*NWG + ng*VWN + 6] = bvec.s6;
         blm[kg*NWG + ng*VWN + 7] = bvec.s7;
      #elif VWN == 16
         blm[kg*NWG + ng*VWN + 0] = bvec.s0;
         blm[kg*NWG + ng*VWN + 1] = bvec.s1;
         blm[kg*NWG + ng*VWN + 2] = bvec.s2;
         blm[kg*NWG + ng*VWN + 3] = bvec.s3;
         blm[kg*NWG + ng*VWN + 4] = bvec.s4;
         blm[kg*NWG + ng*VWN + 5] = bvec.s5;
         blm[kg*NWG + ng*VWN + 6] = bvec.s6;
         blm[kg*NWG + ng*VWN + 7] = bvec.s7;
         blm[kg*NWG + ng*VWN + 8] = bvec.s8;
         blm[kg*NWG + ng*VWN + 9] = bvec.s9;
         blm[kg*NWG + ng*VWN + 10] = bvec.sA;
         blm[kg*NWG + ng*VWN + 11] = bvec.sB;
         blm[kg*NWG + ng*VWN + 12] = bvec.sC;
         blm[kg*NWG + ng*VWN + 13] = bvec.sD;
         blm[kg*NWG + ng*VWN + 14] = bvec.sE;
         blm[kg*NWG + ng*VWN + 15] = bvec.sF;
      #endif
      if (b_conjugate) {
        for (int vn=0; vn<VWN; ++vn) {
          COMPLEX_CONJUGATE(blm[kg*NWG + ng*VWN + vn]);
        }
      }
    }
  }
}

// =================================================================================================

// Caches on-chip local memory into per-thread private memory (registers). This function is specific
// for caching the A input matrix.
inline void LocalToPrivateDirectA(__local real* alm, real apm[MWI], const int kg,
                                  const int a_transpose) {
  #pragma unroll
  for (int mi=0; mi<MWI; ++mi) {
    const int mg = mi + get_local_id(0)*MWI;
    const int index = (a_transpose) ? mg*KWG + kg : kg*MWG + mg;
    apm[mi] = alm[index];
  }
}

// Same as above, but now for the B input matrix
inline void LocalToPrivateDirectB(__local real* blm, real bpm[NWI], const int kg,
                                  const int b_transpose) {
  #pragma unroll
  for (int ni=0; ni<NWI; ++ni) {
    const int ng = ni + get_local_id(1)*NWI;
    const int index = (b_transpose) ? ng*KWG + kg : kg*NWG + ng;
    bpm[ni] = blm[index];
  }
}

// =================================================================================================

// Initializes the accumulation registers to zero
inline void InitAccRegistersDirect(real cpm[NWI][MWI]) {
  #pragma unroll
  for (int mi=0; mi<MWI; ++mi) {
    #pragma unroll
    for (int ni=0; ni<NWI; ++ni) {
      SetToZero(cpm[ni][mi]);
    }
  }
}

// =================================================================================================

// Performs the actual computation: Cpm += Apm * Bpm
inline void MultiplyAccumulateDirect(real cpm[NWI][MWI], real apm[MWI], real bpm[NWI]) {
  #pragma unroll
  for (int ni=0; ni<NWI; ++ni) {
    #pragma unroll
    for (int mi=0; mi<MWI; ++mi) {
      MultiplyAdd(cpm[ni][mi], apm[mi], bpm[ni]);
    }
  }
}

// =================================================================================================

// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication
// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm
inline void StoreResultsDirect(__global real* cgm, real cpm[NWI][MWI],
                               const int kSizeM, const int kSizeN,
                               const real alpha, const real beta,
                               const int c_ld, const int c_offset, const int c_transpose) {
  #pragma unroll
  for (int ni=0; ni<NWI; ++ni) {
    #pragma unroll
    for (int mi=0; mi<MWI; ++mi) {
      int mg = mi + get_local_id(0)*MWI;
      int ng = ni + get_local_id(1)*NWI;
      int idm = mg + GetGroupID0() * MWG;
      int idn = ng + GetGroupID1() * NWG;

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
__attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))
__kernel void XgemmDirect(const int kSizeM, const int kSizeN, const int kSizeK,
                          const real_arg arg_alpha,
                          const real_arg arg_beta,
                          const __global realM* restrict agm, const int a_offset, const int a_ld,
                          const __global realN* restrict bgm, const int b_offset, const int b_ld,
                          __global real* cgm, const int c_offset, const int c_ld,
                          const int a_transpose, const int b_transpose, const int c_transpose,
                          const int a_conjugate, const int b_conjugate) {
  const real alpha = GetRealArg(arg_alpha);
  const real beta = GetRealArg(arg_beta);

  // Extra pointers to scalar versions of global memory
  const __global real* restrict agms = (const __global real* restrict) agm;
  const __global real* restrict bgms = (const __global real* restrict) bgm;

  // Allocates workgroup-private memory (local memory)
  __local real alm[KWG * MWG];
  __local real blm[KWG * NWG];

  // Combined thread identifier (volatile to disable caching)
  volatile int tid = get_local_id(0) + MDIMC*get_local_id(1);

  // Allocates workitem-private memory (registers)
  real apm[MWI];
  real bpm[NWI];
  real cpm[NWI][MWI];

  // Initializes the accumulation registers
  InitAccRegistersDirect(cpm);

  // The faster version of GEMM is not allowed on the (incomplete) borders. Therefore, this section
  // processes only the main parts: output blocks of MWG by NWG.
  const int idm = get_local_id(0) * MWI + GetGroupID0() * MWG;
  const int idn = get_local_id(1) * NWI + GetGroupID1() * NWG;
  if ((idm < (kSizeM/MWG)*MWG) && (idn < (kSizeN/NWG)*NWG) &&
      (a_ld % VWM == 0) && (b_ld % VWN == 0)) {

    // Loops over all complete workgroup tiles
    int kwg = 0;
    for (; kwg < (kSizeK/KWG) * KWG; kwg+=KWG) {

      // Loads data: off-chip --> local (matrix A and B)
      GlobalToLocalDirectA(agm, alm, a_ld, a_offset, tid, kwg, a_transpose, a_conjugate);
      GlobalToLocalDirectB(bgm, blm, b_ld, b_offset, tid, kwg, b_transpose, b_conjugate);
      barrier(CLK_LOCAL_MEM_FENCE);

      // Loops over all workitem tiles, unrolled by a factor KWI
      for (int pwi=0; pwi<KWG; pwi+=KWI) {
        #pragma unroll
        for (int pit=0; pit<KWI; ++pit) {
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
      for (int mi=0; mi<MWI; ++mi) {
        const int a_index = (a_transpose) ? (idm + mi)*a_ld + idk : idk*a_ld + (idm + mi);
        apm[mi] = agms[a_index + a_offset];
        if (a_conjugate) { COMPLEX_CONJUGATE(apm[mi]); }
      }

      // Loads B into register memory
      #pragma unroll
      for (int ni=0; ni<NWI; ++ni) {
        const int b_index = (b_transpose) ? (idn + ni)*b_ld + idk : idk*b_ld + (idn + ni);
        bpm[ni] = bgms[b_index + b_offset];
        if (b_conjugate) { COMPLEX_CONJUGATE(bpm[ni]); }
      }

      // Performs the accumulation (Cpm += Apm * Bpm)
      MultiplyAccumulateDirect(cpm, apm, bpm);
    }

    #if GLOBAL_MEM_FENCE == 1
      barrier(CLK_GLOBAL_MEM_FENCE);
    #endif

    // Stores a tile of results and performs the multiplication with alpha and beta
    StoreResultsDirect(cgm, cpm, kSizeM, kSizeN, alpha, beta, c_ld, c_offset, c_transpose);
  }

  // Simple but slow version for the parts on the edge (incomplete tiles in M and N-dimensions)
  else {

    // Loop over the K-dimension
    for (int idk = 0; idk < kSizeK; ++idk) {

      // Loads A into register memory
      #pragma unroll
      for (int mi=0; mi<MWI; ++mi) {
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
      for (int ni=0; ni<NWI; ++ni) {
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
    for (int ni=0; ni<NWI; ++ni) {
      #pragma unroll
      for (int mi=0; mi<MWI; ++mi) {
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
