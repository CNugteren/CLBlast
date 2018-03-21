
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
#if GEMMK == 0

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

  // Allocates workitem-private memory (registers)
  #pragma promote_to_registers
  realM apm[MWI/VWM];
  #pragma promote_to_registers
  realN bpm[NWI/VWN];
  #pragma promote_to_registers
  realM cpm[NWI*(MWI/VWM)];

  // Combined thread identifier (volatile to disable caching)
  #if SA == 1 || SB == 1
    volatile int tid = get_local_id(0) + MDIMC*get_local_id(1);
  #endif

  // Initializes the accumulation registers
  #pragma unroll
  for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
    #pragma unroll
    for (int _ni = 0; _ni < NWI; _ni += 1) {
      cpm[_ni * (MWI/VWM) + _mi] = InitAccRegisters();
    }
  }


  // Loops over all workgroup tiles
  for (int kwg = 0; kwg < kSizeK; kwg += KWG) {

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
    for (int pwi = 0; pwi < KWG; pwi += KWI) {
      #pragma unroll
      for (int _pit = 0; _pit < KWI; _pit += 1) {
        #if SA == 0 || SB == 0
          int idk = kwg + pwi + _pit;
        #endif
        #if SA == 1 || SB == 1
          int kg = pwi + _pit;
        #endif

        #pragma unroll
        for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
          // Loads data: local --> private (matrix A)
          #if SA == 1
            apm[_mi] = LocalToPrivateA(alm, _mi, kg);
          // Loads data: off-chip --> private (matrix A)
          #else
            apm[_mi] = GlobalToPrivateA(agm, _mi, kSizeM, idk, kwg);
          #endif
        }

        // Loads data: local --> private (matrix B)
        #pragma unroll
        for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
          #if SB == 1
            bpm[_ni] = LocalToPrivateB(blm, _ni, kg);
          // Loads data: off-chip --> private (matrix B)
          #else
            bpm[_ni] = GlobalToPrivateB(bgm, _ni, kSizeN, idk);
          #endif
        }

        // Performs the accumulation (Cpm += Apm * Bpm)
        #pragma unroll
        for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
          #pragma unroll
          for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
            const realM aval = apm[_mi];
            #if VWN == 1
              cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[_ni]);
            #elif VWN == 2
              cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[_ni].x);
              cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi], aval, bpm[_ni].y);
            #elif VWN == 4
              cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[_ni].x);
              cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi], aval, bpm[_ni].y);
              cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi], aval, bpm[_ni].z);
              cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi], aval, bpm[_ni].w);
            #elif VWN == 8
              cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[_ni].s0);
              cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi], aval, bpm[_ni].s1);
              cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi], aval, bpm[_ni].s2);
              cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi], aval, bpm[_ni].s3);
              cpm[(_ni*VWN + 4)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 4)*(MWI/VWM) + _mi], aval, bpm[_ni].s4);
              cpm[(_ni*VWN + 5)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 5)*(MWI/VWM) + _mi], aval, bpm[_ni].s5);
              cpm[(_ni*VWN + 6)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 6)*(MWI/VWM) + _mi], aval, bpm[_ni].s6);
              cpm[(_ni*VWN + 7)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 7)*(MWI/VWM) + _mi], aval, bpm[_ni].s7);
            #elif VWN == 16
              cpm[(_ni*VWN + 0 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0 )*(MWI/VWM) + _mi], aval, bpm[_ni].s0);
              cpm[(_ni*VWN + 1 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1 )*(MWI/VWM) + _mi], aval, bpm[_ni].s1);
              cpm[(_ni*VWN + 2 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 2 )*(MWI/VWM) + _mi], aval, bpm[_ni].s2);
              cpm[(_ni*VWN + 3 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 3 )*(MWI/VWM) + _mi], aval, bpm[_ni].s3);
              cpm[(_ni*VWN + 4 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 4 )*(MWI/VWM) + _mi], aval, bpm[_ni].s4);
              cpm[(_ni*VWN + 5 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 5 )*(MWI/VWM) + _mi], aval, bpm[_ni].s5);
              cpm[(_ni*VWN + 6 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 6 )*(MWI/VWM) + _mi], aval, bpm[_ni].s6);
              cpm[(_ni*VWN + 7 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 7 )*(MWI/VWM) + _mi], aval, bpm[_ni].s7);
              cpm[(_ni*VWN + 8 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 8 )*(MWI/VWM) + _mi], aval, bpm[_ni].s8);
              cpm[(_ni*VWN + 9 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 9 )*(MWI/VWM) + _mi], aval, bpm[_ni].s9);
              cpm[(_ni*VWN + 10)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 10)*(MWI/VWM) + _mi], aval, bpm[_ni].sA);
              cpm[(_ni*VWN + 11)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 11)*(MWI/VWM) + _mi], aval, bpm[_ni].sB);
              cpm[(_ni*VWN + 12)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 12)*(MWI/VWM) + _mi], aval, bpm[_ni].sC);
              cpm[(_ni*VWN + 13)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 13)*(MWI/VWM) + _mi], aval, bpm[_ni].sD);
              cpm[(_ni*VWN + 14)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 14)*(MWI/VWM) + _mi], aval, bpm[_ni].sE);
              cpm[(_ni*VWN + 15)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 15)*(MWI/VWM) + _mi], aval, bpm[_ni].sF);
            #endif
          }
        }

      }
    }
    #if SA == 1 || SB == 1
      barrier(CLK_LOCAL_MEM_FENCE);
    #endif
  }
  #if GLOBAL_MEM_FENCE == 1
    barrier(CLK_GLOBAL_MEM_FENCE);
  #endif

  // Stores an MWG * NWG tile of results and performs the multiplication with alpha and beta
  StoreResults(cgm, cpm, kSizeM, alpha, beta);
}

// =================================================================================================
#else  // GEMMK == 1

// Following is a Qualcomm specific kernel
// https://developer.qualcomm.com/blog/matrix-multiply-adreno-gpus-part-2-host-code-and-kernel
// Assumes:
// SA == 0, SB == 0
// VWN == MWI == KWG
// VWN <= 4, VWM <= 4
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

  // Loops over all workgroup tiles
  for (int kwg = 0; kwg < kSizeK; kwg += KWG) {

    // Loads the B matrix from global memory and stores into registers
    #pragma unroll
    for (int _mi = 0; _mi < VWM; _mi += 1) {
      const int b_index = (kwg + _mi) * kSizeN + tidx * MWI;
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
      const int a_index = (tidy * NWI + _ni) * kSizeK + kwg;
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

#endif // GEMMK
// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
