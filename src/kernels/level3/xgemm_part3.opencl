
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

  const __global real* restrict A = (const __global real* restrict) &agm[0];
  const __global real* restrict B = (const __global real* restrict) &bgm[0];

  // Allocates workitem-private memory (registers)
  #pragma promote_to_registers
  realN apm[NWI];
  #pragma promote_to_registers
  realM bpm[MWI];
  #pragma promote_to_registers
  realM cpm[NWI*(MWI/VWM)];

  if (((gx << 2) < kSizeN) && ((gy << 3) < kSizeM)) {

    #pragma unroll
    for (int _ni = 0; _ni < NWI; _ni += 1) {
      SetToZero(cpm[_ni]);
    }

    const int A_y_off = (gy << 3) * kSizeK;

    for (int pos = 0; pos < kSizeK; pos += 4) {
      #pragma unroll
      for (int i = 0; i < 4; i++) {
        // Original code: b[i] = read_imagef(Bi, (int2)(gx, pos + i));
        int B_off = (pos + i) * kSizeN + (gx << 2);
        bpm[i] = vload4(0, B + B_off);
      }

      int A_off = A_y_off + pos;

      #pragma unroll
      for (int i = 0; i < 8; i++) {
        apm[i] = vload4(0, A + A_off);
        A_off += kSizeK;
      }

      #pragma unroll
      for (int i = 0; i < 8; i++) {
        cpm[i] += apm[i].x * bpm[0] + apm[i].y * bpm[1] + apm[i].z * bpm[2] + apm[i].w * bpm[3];
      }

      #if SA == 1 || SB == 1
        barrier(CLK_LOCAL_MEM_FENCE);
      #endif
    }
    #if GLOBAL_MEM_FENCE == 1
      barrier(CLK_GLOBAL_MEM_FENCE);
    #endif

    // Stores an MWG * NWG tile of results and performs the multiplication with alpha and beta
    StoreResults(cgm, cpm, kSizeN, alpha, beta);
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
