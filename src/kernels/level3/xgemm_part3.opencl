
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 3 of 4 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Following is a Qualcomm specific kernel
// https://developer.qualcomm.com/blog/matrix-multiply-adreno-gpus-part-2-host-code-and-kernel
// Assumes:
// SA == 0, SB == 0
// VWN == MWI
// KWG == 4
// Unused: KWI, STRM, STRN
INLINE_FUNC void XgemmBody(const int kSizeM, const int kSizeN, const int kSizeK,
                           const __global realM* restrict agm, const __global realN* restrict bgm,
                           __global realM* cgm, const real alpha, const real beta) {
  const int tidx = get_global_id(0);
  const int tidy = get_global_id(1);

  const __global real* restrict a_ptr = (const __global real* restrict) &agm[0];
  const __global real* restrict b_ptr = (const __global real* restrict) &bgm[0];

  // Allocates workitem-private memory (registers)
  #pragma promote_to_registers
  realN apm[NWI];
  #pragma promote_to_registers
  realM bpm[MWI];
  #pragma promote_to_registers
  realM cpm[NWI*(MWI/VWM)];

  #pragma unroll
  for (int _ni = 0; _ni < NWI; _ni += 1) {
    SetToZero(cpm[_ni]);
  }

  for (int _ki = 0; _ki < kSizeK; _ki += VWN) {

    // Loads the B matrix from global memory and stores into registers
    #pragma unroll
    for (int _mi = 0; _mi < VWM; _mi += 1) {
      const int b_index = (_ki + _mi) * kSizeN + tidx * MWI;
      #if VWM == 1
        bpm[_mi] = b_ptr[b_index];
      #elif VWM == 2
        bpm[_mi] = vload2(0, b_ptr + b_index);
      #elif VWM == 4
        bpm[_mi] = vload4(0, b_ptr + b_index);
      #endif
    }

    // Loads the A matrix from global memory and stores into registers
    #pragma unroll
    for (int _ni = 0; _ni < NWI; _ni += 1) {
      const int a_index = (tidy * NWI + _ni) * kSizeK + _ki;
      #if VWN == 1
        apm[_ni] = a_ptr[a_index];
      #elif VWN == 2
        apm[_ni] = vload2(0, a_ptr + a_index);
      #elif VWN == 4
        apm[_ni] = vload4(0, a_ptr + a_index);
      #endif
    }

    // Performs the accumulation (Cpm += Apm * Bpm)
    #pragma unroll
    for (int _ni = 0; _ni < NWI; _ni += 1) {
      #if VWN == 1
        cpm[_ni] += apm[_ni] * bpm[0];
      #elif VWN == 2
        cpm[_ni] += apm[_ni].x * bpm[0] + apm[_ni].y * bpm[1];
      #elif VWN == 4
        cpm[_ni] += apm[_ni].x * bpm[0] + apm[_ni].y * bpm[1] + apm[_ni].z * bpm[2] + apm[_ni].w * bpm[3];
      #endif
    }
  }

  // Stores an MWG * NWG tile of results and performs the multiplication with alpha and beta
  StoreResults(cgm, cpm, kSizeN, alpha, beta);
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
