
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 2 of 2 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

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
      int idm = mg + GetGroupID0() * WGD;
      int idn = ng + GetGroupID1() * WGD;

      // Determines the destination index
      const int c_index = (c_transpose) ? idm*c_ld + idn : idn*c_ld + idm;

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

// =================================================================================================

// Main body of the kernel. This is the direct version without restrictions.
inline void XgemmDirect(const int kSizeM, const int kSizeN, const int kSizeK,
                        const real_arg arg_alpha,
                        const real_arg arg_beta,
                        const __global realMD* restrict agm, const int a_offset, const int a_ld,
                        const __global realND* restrict bgm, const int b_offset, const int b_ld,
                        __global real* cgm, const int c_offset, const int c_ld,
                        __local real* alm, __local real* blm,
                        const int a_transpose, const int b_transpose, const int c_transpose,
                        const int a_conjugate, const int b_conjugate) {
  const real alpha = GetRealArg(arg_alpha);
  const real beta = GetRealArg(arg_beta);

  // Extra pointers to scalar versions of global memory
  const __global real* restrict agms = (const __global real* restrict) agm;
  const __global real* restrict bgms = (const __global real* restrict) bgm;

  // Allocates workitem-private memory (registers)
  real apm[MWID];
  real bpm[NWID];
  real cpm[NWID][MWID];

  // Initializes the accumulation registers
  InitAccRegistersDirect(cpm);

  // The faster version of GEMM is not allowed on the (incomplete) borders. Therefore, this section
  // processes only the main parts: output blocks of WGD by WGD.
  const int idm = get_local_id(0) * MWID + GetGroupID0() * WGD;
  const int idn = get_local_id(1) * NWID + GetGroupID1() * WGD;
  if ((idm < (kSizeM/WGD)*WGD) && (idn < (kSizeN/WGD)*WGD) &&
      (a_ld % VWMD == 0) && (b_ld % VWND == 0)) {

    // Loops over all complete workgroup tiles
    int kwg = 0;
    for (; kwg < (kSizeK/WGD) * WGD; kwg+=WGD) {

      // Loads data: off-chip --> local (matrix A and B)
      GlobalToLocalDirectA(agm, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate);
      GlobalToLocalDirectB(bgm, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate);
      barrier(CLK_LOCAL_MEM_FENCE);

      // Loops over all workitem tiles, unrolled by a factor KWID
      for (int pwi=0; pwi<WGD; pwi+=KWID) {
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

// Direct version of the GEMM kernel with [A, B] = [non-transposed, non-transposed]
__attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
__kernel void XgemmDirectNN(const int kSizeM, const int kSizeN, const int kSizeK,
                            const real_arg arg_alpha, const real_arg arg_beta,
                            const __global realMD* restrict agm, const int a_offset, const int a_ld,
                            const __global realND* restrict bgm, const int b_offset, const int b_ld,
                            __global real* cgm, const int c_offset, const int c_ld,
                            const int c_transpose, const int a_conjugate, const int b_conjugate) {
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 0, 0, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the GEMM kernel with [A, B] = [non-transposed, transposed]
__attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
__kernel void XgemmDirectNT(const int kSizeM, const int kSizeN, const int kSizeK,
                            const real_arg arg_alpha, const real_arg arg_beta,
                            const __global realMD* restrict agm, const int a_offset, const int a_ld,
                            const __global realND* restrict bgm, const int b_offset, const int b_ld,
                            __global real* cgm, const int c_offset, const int c_ld,
                            const int c_transpose, const int a_conjugate, const int b_conjugate) {
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 0, 1, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the GEMM kernel with [A, B] = [transposed, non-transposed]
__attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
__kernel void XgemmDirectTN(const int kSizeM, const int kSizeN, const int kSizeK,
                            const real_arg arg_alpha, const real_arg arg_beta,
                            const __global realMD* restrict agm, const int a_offset, const int a_ld,
                            const __global realND* restrict bgm, const int b_offset, const int b_ld,
                            __global real* cgm, const int c_offset, const int c_ld,
                            const int c_transpose, const int a_conjugate, const int b_conjugate) {
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 1, 0, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the GEMM kernel with [A, B] = [transposed, transposed]
__attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
__kernel void XgemmDirectTT(const int kSizeM, const int kSizeN, const int kSizeK,
                            const real_arg arg_alpha, const real_arg arg_beta,
                            const __global realMD* restrict agm, const int a_offset, const int a_ld,
                            const __global realND* restrict bgm, const int b_offset, const int b_ld,
                            __global real* cgm, const int c_offset, const int c_ld,
                            const int c_transpose, const int a_conjugate, const int b_conjugate) {
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 1, 1, c_transpose, a_conjugate, b_conjugate);
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
