
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

// Main body of the matrix-multiplication algorithm. It calls various (inlined) functions.
INLINE_FUNC void XgemmBody(const int kSizeM, const int kSizeN, const int kSizeK,
                           const __global realM* restrict agm, const __global realN* restrict bgm,
                           __global realM* cgm, const real alpha, const real beta
                           #if SA == 1 && SB == 1
                             , LOCAL_PTR realM* alm, LOCAL_PTR realN* blm
                           #elif SA == 1
                             , LOCAL_PTR realM* alm
                           #elif SB == 1
                             , LOCAL_PTR realN* blm
                           #endif
                           ) {
  //
  // Following is a Qualcomm specific kernel to test, it assumes alpha=1.0 and beta=0.0
  // https://developer.qualcomm.com/blog/matrix-multiply-adreno-gpus-part-2-host-code-and-kernel
  //

  const int gx = get_global_id(0);
  const int gy = get_global_id(1);

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
    #pragma unroll
    for (int _mi = 0; _mi < VWM; _mi += 1) {
      const int b_index = (_ki + _mi) * kSizeN + gx * MWI;
      bpm[_mi] = vload4(0, b_ptr + b_index);
    }

    // Loads the data from global memory and stores into registers
    #pragma unroll
    for (int _ni = 0; _ni < NWI; _ni += 1) {
      const int a_index = (gy * NWI + _ni) * kSizeK + _ki;
      apm[_ni] = vload4(0, a_ptr + a_index);
    }

    // Performs the accumulation (Cpm += Apm * Bpm)
    #pragma unroll
    for (int _ni = 0; _ni < NWI; _ni += 1) {
      // VWN == MWI
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
