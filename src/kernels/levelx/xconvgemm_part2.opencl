
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the an implementation of 3D convolution on a 4D image using GEMM kernels. It
// uses parameters from the direct GEMM kernel. This part contains the main kernel (2/2).
// This uses "CONVGEMM_WITH_IM2COL" as a switch to select between direct convgemm or first running
// the im2col kernel to create a 'col' temporary matrix.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================
#if defined(ROUTINE_CONVGEMM)

// ConvGEMM kernel
#if defined(CONVGEMM_WITH_IM2COL)
__kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
void Xconvgemm(const int num_patches, const int num_kernels, const int patch_size,
               const __global realND* restrict kernelgm, const int kernel_offset,
               __global real* resultgm, const int result_offset, const int result_stride,
               const __global realMD* restrict colgm, const int col_offset, const int col_stride)
#else
INLINE_FUNC void Xconvgemm(const int num_patches, const int num_kernels, const int patch_size,
                           const __global realND* restrict kernelgm, const int kernel_offset,
                           __global real* resultgm, const int result_offset, const int result_stride,
                           const __global realMD* restrict imagegm, const int image_offset,
                           const int input_h, const int input_w, const int channels,
                           const int kernel_h, const int kernel_w,
                           const int pad_h, const int pad_w,
                           const int stride_h, const int stride_w,
                           const int dilation_h, const int dilation_w,
                           const int output_h, const int output_w,
                           LOCAL_PTR real* alm, LOCAL_PTR real* blm,
                           const bool kernel_flip)
#endif
{

  // Batch offsets
  const int batch = get_group_id(2);
  #if defined(CONVGEMM_WITH_IM2COL)
    const int col_offset_batch = col_offset + col_stride * batch;
  #else
    const int image_offset_batch = image_offset + channels * input_h * input_w * batch;
  #endif
  const int result_offset_batch = result_offset + result_stride * batch;

#if defined(CONVGEMM_WITH_IM2COL)
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
#endif

  // Extra pointers to scalar versions of global memory
  #if defined(CONVGEMM_WITH_IM2COL)
    const __global real* restrict colgms = (const __global real* restrict) colgm;
  #else
    const __global real* restrict imagegms = (const __global real* restrict) imagegm;
  #endif
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

  // Global m/n indices
  const int idm = get_local_id(0) * MWID + GetGroupID0() * WGD;
  const int idn = get_local_id(1) * NWID + GetGroupID1() * WGD;
  #if !defined(CONVGEMM_WITH_IM2COL)
    const int w_id = idm % output_w;
    const int h_id = idm / output_w;
  #endif

  // The faster version of GEMM is not allowed on the (incomplete) borders. Therefore, this section
  // processes only the main parts: output blocks of WGD by WGD.
  if ((idm < (num_patches/WGD)*WGD) && (idn < (num_kernels/WGD)*WGD)) {

    // Loops over all complete workgroup tiles (K-dimension)
    int kwg = 0;
    for (; kwg < (patch_size/WGD) * WGD; kwg += WGD) {

      // Loads data: off-chip --> local (matrix A and B)
      #if defined(CONVGEMM_WITH_IM2COL)
        if (num_patches % VWMD == 0 && col_offset_batch % VWMD == 0) {
          GlobalToLocalDirectA(colgm, alm, num_patches, col_offset_batch, kwg, false, false);
        }
        else {
          GlobalToLocalScalarA(colgms, alm, num_patches, col_offset_batch, kwg, false, false);
        }
      #else
        GlobalToLocalCheckedImage(imagegms, alm, image_offset_batch, output_w, kwg,
                                  input_h, input_w, channels, kernel_h, kernel_w,
                                  pad_h, pad_w, stride_h, stride_w,
                                  dilation_h, dilation_w, kernel_flip);
      #endif
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
        #if defined(CONVGEMM_WITH_IM2COL)
          apd[_mi] = GlobalToPrivateDirectA(colgms, _mi, num_patches, col_offset_batch, idm, kwg, false, false);
        #else
          const int w_id = (idm + _mi) % output_w;
          const int h_id = (idm + _mi) / output_w;
          apd[_mi] = GlobalToPrivateCheckedImage(imagegms, image_offset_batch, h_id, w_id, kwg,
                                                 input_h, input_w, channels, kernel_h, kernel_w,
                                                 pad_h, pad_w, stride_h, stride_w,
                                                 dilation_h, dilation_w, kernel_flip);
        #endif
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

      // Loads data: off-chip --> local
      #if defined(CONVGEMM_WITH_IM2COL)
        GlobalToLocalCheckedA(colgms, alm, num_patches, col_offset_batch, kwg, false, false, num_patches, patch_size);
      #else
        GlobalToLocalCheckedImage(imagegms, alm, image_offset_batch, output_w, kwg,
                                  input_h, input_w, channels, kernel_h, kernel_w,
                                  pad_h, pad_w, stride_h, stride_w,
                                  dilation_h, dilation_w, kernel_flip);
      #endif
      GlobalToLocalCheckedB(kernelgms, blm, patch_size, kernel_offset, kwg, true, false, num_kernels, patch_size);
      barrier(CLK_LOCAL_MEM_FENCE);

      // Loops over all workitem tiles, unrolled by a factor KWID
      for (int pwi = 0; pwi < WGD; pwi += KWID) {
        #pragma unroll
        for (int _pit = 0; _pit < KWID; _pit += 1) {
          int kg = pwi + _pit;

          // Loads data: local --> private
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

      // Loads data: off-chip --> private
      #pragma unroll
      for (int _mi = 0; _mi < MWID; _mi += 1) {
      #if defined(CONVGEMM_WITH_IM2COL)
        apd[_mi] = GlobalToPrivateCheckedA(colgms, _mi, num_patches, col_offset_batch, idm, kwg, false, false, num_patches);
      #else
        const int w_id = (idm + _mi) % output_w;
        const int h_id = (idm + _mi) / output_w;
        apd[_mi] = GlobalToPrivateCheckedImage(imagegms, image_offset_batch, h_id, w_id, kwg,
                                               input_h, input_w, channels, kernel_h, kernel_w,
                                               pad_h, pad_w, stride_h, stride_w,
                                               dilation_h, dilation_w, kernel_flip);
      #endif
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

#if !defined(CONVGEMM_WITH_IM2COL)
__kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
void XconvgemmFlip(const int num_patches, const int num_kernels, const int patch_size,
                   const __global realND* restrict kernelgm, const int kernel_offset,
                   __global real* resultgm, const int result_offset, const int result_stride,
                   const __global realMD* restrict imagegm, const int image_offset,
                   const int input_h, const int input_w, const int channels,
                   const int kernel_h, const int kernel_w,
                   const int pad_h, const int pad_w,
                   const int stride_h, const int stride_w,
                   const int dilation_h, const int dilation_w,
                   const int output_h, const int output_w) {
  const bool kernel_flip = true;
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  Xconvgemm(num_patches, num_kernels, patch_size,
            kernelgm, kernel_offset, resultgm, result_offset, result_stride,
            imagegm, image_offset, input_h, input_w, channels, kernel_h, kernel_w,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
            output_h, output_w, alm, blm, kernel_flip);
}

__kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
void XconvgemmNormal(const int num_patches, const int num_kernels, const int patch_size,
                     const __global realND* restrict kernelgm, const int kernel_offset,
                     __global real* resultgm, const int result_offset, const int result_stride,
                     const __global realMD* restrict imagegm, const int image_offset,
                     const int input_h, const int input_w, const int channels,
                     const int kernel_h, const int kernel_w,
                     const int pad_h, const int pad_w,
                     const int stride_h, const int stride_w,
                     const int dilation_h, const int dilation_w,
                     const int output_h, const int output_w) {
  const bool kernel_flip = false;
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  Xconvgemm(num_patches, num_kernels, patch_size,
            kernelgm, kernel_offset, resultgm, result_offset, result_stride,
            imagegm, image_offset, input_h, input_w, channels, kernel_h, kernel_w,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
            output_h, output_w, alm, blm, kernel_flip);
}

#endif  // !defined(CONVGEMM_WITH_IM2COL)

#endif  // defined(ROUTINE_CONVGEMM)

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
