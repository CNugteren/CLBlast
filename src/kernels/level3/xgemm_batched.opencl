
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the batched version of the non-direct GEMM kernel. See part 1 for information
// about the non-batched version of the kernel.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================
#if defined(ROUTINE_GEMMBATCHED)

#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))
#endif
void XgemmBatched(const int kSizeM, const int kSizeN, const int kSizeK,
                  const __constant real_arg* arg_alphas,
                  const __constant real_arg* arg_betas,
                  const __global realM* restrict agm, const int a_one, const int a_two,
                  const __global realN* restrict bgm, const int b_one, const int b_two,
                  __global realM* cgm, const int c_one, const int c_two) {
  const int batch = get_group_id(2);
  const real alpha = GetRealArg(arg_alphas[batch]);
  const real beta = GetRealArg(arg_betas[batch]);

  // Sets the offsets
  const int a_offset = batch * a_one * a_two;
  const int b_offset = batch * b_one * b_two;
  const int c_offset = batch * c_one * c_two;
  const __global realM* restrict agm_ = &agm[a_offset / VWM];
  const __global realN* restrict bgm_ = &bgm[b_offset / VWN];
  __global realM* restrict cgm_ = &cgm[c_offset / VWM];

  // Allocates workgroup-private memory (local memory)
  #if SA == 1
    __local realM alm[KWG * MWG/VWM];
  #endif
  #if SB == 1
    __local realN blm[KWG * NWG/VWN];
  #endif

  // Computes the matrix-multiplication and stores the result in global memory
  #if SA == 1 && SB == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, alpha, beta, alm, blm);
  #elif SA == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, alpha, beta, alm);
  #elif SB == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, alpha, beta, blm);
  #else
    XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, alpha, beta);
  #endif
}

#endif
// =================================================================================================
#if defined(ROUTINE_GEMMSTRIDEDBATCHED)

#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))
#endif
void XgemmStridedBatched(const int kSizeM, const int kSizeN, const int kSizeK,
                         const real_arg arg_alpha, const real_arg arg_beta,
                         const __global realM* restrict agm, const int a_one, const int a_two,
                         const __global realN* restrict bgm, const int b_one, const int b_two,
                         __global realM* cgm, const int c_one, const int c_two) {
  const int batch = get_group_id(2);
  const real alpha = GetRealArg(arg_alpha);
  const real beta = GetRealArg(arg_beta);

  // Sets the offsets
  const int a_offset = batch * a_one * a_two;
  const int b_offset = batch * b_one * b_two;
  const int c_offset = batch * c_one * c_two;
  const __global realM* restrict agm_ = &agm[a_offset / VWM];
  const __global realN* restrict bgm_ = &bgm[b_offset / VWN];
  __global realM* restrict cgm_ = &cgm[c_offset / VWM];

  // Allocates workgroup-private memory (local memory)
  #if SA == 1
    __local realM alm[KWG * MWG/VWM];
  #endif
  #if SB == 1
    __local realN blm[KWG * NWG/VWN];
  #endif

  // Computes the matrix-multiplication and stores the result in global memory
  #if SA == 1 && SB == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, alpha, beta, alm, blm);
  #elif SA == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, alpha, beta, alm);
  #elif SB == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, alpha, beta, blm);
  #else
    XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, alpha, beta);
  #endif
}

#endif
// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
