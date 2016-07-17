
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

  // Allocates workitem-private memory (registers)
  real apm[MWI];
  real bpm[NWI];
  real cpm[NWI][MWI];

  // Initializes the accumulation registers
  InitAccRegistersDirect(cpm);

  // The faster version of GEMM is not allowed on the (incomplete) borders. Therefore, this section
  // processes only the main parts: output blocks of NWI by MWI.
  const int idm = get_local_id(0) * MWI + GetGroupID0() * MWG;
  const int idn = get_local_id(1) * NWI + GetGroupID1() * NWG;
  if ((idm < kSizeM - MWI) && (idn < kSizeN - NWI)) {

    // Loops over all complete workgroup tiles
    int kwg = 0;
    // TODO: Implement a faster version with local memory and vector loads
    // for (; kwg < kSizeK - KWG; kwg+=KWG) { }

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
