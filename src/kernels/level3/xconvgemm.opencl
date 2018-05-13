
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the an implementation of 3D convolution on a 4D image using GEMM kernels. It
// uses parameters from the direct GEMM kernel.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================
#if defined(ROUTINE_CONVGEMM)

// ConvGEMM kernel
__kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
void Xconvgemm(const int num_patches, const int num_kernels, const int patch_size,
               const __global realMD* restrict colgm, const int col_offset, const int col_stride,
               const __global realND* restrict kernelgm, const int kernel_offset,
               __global real* resultgm, const int result_offset, const int result_stride) {

  // Batch offsets
  const int batch = get_group_id(2);
  const int col_offset_batch = col_offset + col_stride * batch;
  const int result_offset_batch = result_offset + result_stride * batch;

  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];

  // Extra pointers to scalar versions of global memory
  const __global real* restrict colgms = (const __global real* restrict) colgm;
  const __global real* restrict kernelgms = (const __global real* restrict) kernelgm;

  // Allocates workitem-private memory (registers)
  #pragma promote_to_registers
  real apd[MWID];
  #pragma promote_to_registers
  real bpd[NWID];
  #pragma promote_to_registers
  real cpd[NWID * MWID];

  // Initializes the accumulation registers
  #pragma unroll
  for (int _mi = 0; _mi < MWID; _mi += 1) {
    #pragma unroll
    for (int _ni = 0; _ni < NWID; _ni += 1) {
      SetToZero(cpd[_ni * MWID + _mi]);
    }
  }

  // The faster version of GEMM is not allowed on the (incomplete) borders. Therefore, this section
  // processes only the main parts: output blocks of WGD by WGD.
  const int idm = get_local_id(0) * MWID + GetGroupID0() * WGD;
  const int idn = get_local_id(1) * NWID + GetGroupID1() * WGD;
  if ((idm < (num_patches/WGD)*WGD) && (idn < (num_kernels/WGD)*WGD)) {

    // Loops over all complete workgroup tiles (K-dimension)
    int kwg = 0;
    for (; kwg < (patch_size/WGD) * WGD; kwg += WGD) {

      // Loads data: off-chip --> local (matrix A and B)
      if (num_patches % VWMD == 0 && col_offset_batch % VWMD == 0) {
        GlobalToLocalDirectA(colgm, alm, num_patches, col_offset_batch, kwg, false, false);
      }
      else {
        GlobalToLocalScalarA(colgms, alm, num_patches, col_offset_batch, kwg, false, false);
      }
      if (patch_size % VWND == 0 && kernel_offset % VWND == 0) {
        GlobalToLocalDirectB(kernelgm, blm, patch_size, kernel_offset, kwg, true, false);
      }
      else {
        GlobalToLocalScalarB(kernelgms, blm, patch_size, kernel_offset, kwg, true, false);
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      // Loops over all workitem tiles, unrolled by a factor KWID
      for (int pwi = 0; pwi < WGD; pwi += KWID) {
        #pragma unroll
        for (int _pit = 0; _pit < KWID; _pit += 1) {
          int kg = pwi + _pit;

          // Loads data: local --> private (matrix A and B)
          #pragma unroll
          for (int _mi = 0; _mi < MWID; _mi += 1) {
            apd[_mi] = LocalToPrivateDirectA(alm, _mi, kg, false);
          }
          #pragma unroll
          for (int _ni = 0; _ni < NWID; _ni += 1) {
            bpd[_ni] = LocalToPrivateDirectB(blm, _ni, kg, true);
          }

          // Performs the accumulation (Cpmd += Apmd * Bpmd)
          #pragma unroll
          for (int _ni = 0; _ni < NWID; _ni += 1) {
            #pragma unroll
            for (int _mi = 0; _mi < MWID; _mi += 1) {
              MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Loop over the remaining part (incomplete tile in K-dimension)
    for (; kwg < patch_size; ++kwg) {

      // Loads data: off-chip --> private (matrix A and B)
      #pragma unroll
      for (int _mi = 0; _mi < MWID; _mi += 1) {
        apd[_mi] = GlobalToPrivateDirectA(colgms, _mi, num_patches, col_offset_batch, idm, kwg, false, false);
      }
      #pragma unroll
      for (int _ni = 0; _ni < NWID; _ni += 1) {
        bpd[_ni] = GlobalToPrivateDirectB(kernelgms, _ni, patch_size, kernel_offset, idn, kwg, true, false);
      }

      // Performs the accumulation (Cpmd += Apmd * Bpmd)
      #pragma unroll
      for (int _ni = 0; _ni < NWID; _ni += 1) {
        #pragma unroll
        for (int _mi = 0; _mi < MWID; _mi += 1) {
          MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
        }
      }
    }

    // Stores a tile of results
    #pragma unroll
    for (int _ni = 0; _ni < NWID; _ni += 1) {
      #pragma unroll
      for (int _mi = 0; _mi < MWID; _mi += 1) {
        StoreResultsDirect(resultgm, cpd[_ni * MWID + _mi], _mi, _ni, idm, idn,
                           ONE, ZERO, num_patches, result_offset_batch, false);
      }
    }
  }

  // Simple but slower version for the parts on the edge (incomplete tiles in M and N-dimensions)
  else {

    // Loops over all complete workgroup tiles (K-dimension)
    int kwg = 0;
    for (; kwg < (patch_size/WGD) * WGD; kwg+=WGD) {

      // Loads data: off-chip --> local (matrix A and B)
      GlobalToLocalCheckedA(colgms, alm, num_patches, col_offset_batch, kwg, false, false, num_patches, patch_size);
      GlobalToLocalCheckedB(kernelgms, blm, patch_size, kernel_offset, kwg, true, false, num_kernels, patch_size);
      barrier(CLK_LOCAL_MEM_FENCE);

      // Loops over all workitem tiles, unrolled by a factor KWID
      for (int pwi = 0; pwi < WGD; pwi += KWID) {
        #pragma unroll
        for (int _pit = 0; _pit < KWID; _pit += 1) {
          int kg = pwi + _pit;

          // Loads data: local --> private (matrix A and B)
          #pragma unroll
          for (int _mi = 0; _mi < MWID; _mi += 1) {
            apd[_mi] = LocalToPrivateDirectA(alm, _mi, kg, false);
          }
          #pragma unroll
          for (int _ni = 0; _ni < NWID; _ni += 1) {
            bpd[_ni] = LocalToPrivateDirectB(blm, _ni, kg, true);
          }

          // Performs the accumulation (C += A * B)
          #pragma unroll
          for (int _ni = 0; _ni < NWID; _ni += 1) {
            #pragma unroll
            for (int _mi = 0; _mi < MWID; _mi += 1) {
              MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Loop over the remaining part (incomplete tile in K-dimension)
    for (; kwg < patch_size; ++kwg) {

      // Loads data: off-chip --> private (matrix A and B)
      #pragma unroll
      for (int _mi = 0; _mi < MWID; _mi += 1) {
        apd[_mi] = GlobalToPrivateCheckedA(colgms, _mi, num_patches, col_offset_batch, idm, kwg, false, false, num_patches);
      }
      #pragma unroll
      for (int _ni = 0; _ni < NWID; _ni += 1) {
        bpd[_ni] = GlobalToPrivateCheckedB(kernelgms, _ni, patch_size, kernel_offset, idn, kwg, true, false, num_kernels);
      }

      // Performs the accumulation (C += A * B)
      #pragma unroll
      for (int _ni = 0; _ni < NWID; _ni += 1) {
        #pragma unroll
        for (int _mi = 0; _mi < MWID; _mi += 1) {
          MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
        }
      }
    }

    // Stores a tile of results
    #pragma unroll
    for (int _ni = 0; _ni < NWID; _ni += 1) {
      #pragma unroll
      for (int _mi = 0; _mi < MWID; _mi += 1) {
        StoreResultsChecked(resultgm, cpd[_ni * MWID + _mi], _mi, _ni, idm, idn, num_patches, num_kernels,
                            ONE, ZERO, num_patches, result_offset_batch, false);
      }
    }
  }
}

#endif
// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
