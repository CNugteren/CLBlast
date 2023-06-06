
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the an implementation of 3D convolution on a 4D image using GEMM kernels. It
// uses parameters from the direct GEMM kernel. This is the part with the loads from memory (1/2).
// This uses "CONVGEMM_WITH_IM2COL" as a switch to select between direct convgemm or first running
// the im2col kernel to create a 'col' temporary matrix.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================
#if defined(ROUTINE_CONVGEMM) && !defined(CONVGEMM_WITH_IM2COL)

// Loads global off-chip memory into thread-private register files. This function is specific for
// loading the image input tensor. This includes a bounds check.
INLINE_FUNC real GlobalToPrivateCheckedImage(const __global real* restrict imagegm, const int image_offset_batch,
                                             const int h_id, const int w_id, const int kwg,
                                             const int input_h, const int input_w, const int channels,
                                             const int kernel_h, const int kernel_w,
                                             const int pad_h, const int pad_w,
                                             const int stride_h, const int stride_w,
                                             const int dilation_h, const int dilation_w,
                                             const bool kernel_flip) {

  // Im2col indices
  const int kernel_2d_index = kwg % (kernel_h * kernel_w);
  const int kw_id = (kernel_flip)
                  ? kernel_w - kernel_2d_index % kernel_w - 1
                  : kernel_2d_index % kernel_w;
  const int kh_id = (kernel_flip)
                  ? kernel_h - kernel_2d_index / kernel_w - 1
                  : kernel_2d_index / kernel_w;
  const int c_id = kwg / (kernel_h * kernel_w);
  const int h_index = -pad_h + kh_id * dilation_h + stride_h * h_id;
  const int w_index = -pad_w + kw_id * dilation_w + stride_w * w_id;

  // With bounds check
  real result;
  if (h_index >= 0 && h_index < input_h &&
      w_index >= 0 && w_index < input_w) {
    const int image_index = w_index + input_w * (h_index + input_h * c_id);
    result = imagegm[image_index + image_offset_batch];
  }
  else {
    SetToZero(result);
  }
  return result;
}

// Loads global off-chip memory into local (shared) memory on-chip. This function is specific for
// loading the image input tensor. This includes a bounds check.
INLINE_FUNC real GlobalToLocalCheckedImage(const __global real* restrict imagegm, LOCAL_PTR real* alm,
                                           const int image_offset_batch,
                                           const int output_w, const int kwg,
                                           const int input_h, const int input_w, const int channels,
                                           const int kernel_h, const int kernel_w,
                                           const int pad_h, const int pad_w,
                                           const int stride_h, const int stride_w,
                                           const int dilation_h, const int dilation_w,
                                           const bool kernel_flip) {
  #if MDIMCD == MDIMAD
    const int la0 = get_local_id(0);
    const int la1 = get_local_id(1);
  #else
    const int tid = get_local_id(0) + MDIMCD*get_local_id(1);
    const int la0 = tid % MDIMAD;
    const int la1 = tid / MDIMAD;
  #endif
  #pragma unroll
  for (int _mia = 0; _mia < MWAD; _mia += 1) {
    #pragma unroll
    for (int _kia = 0; _kia < KWAD; _kia += 1) {

      // Computes the indices for the global memory
      int mg = _mia + la0*MWAD;
      int kg = _kia + la1*KWAD;
      int idm = mg + GetGroupID0()*WGD;
      int idk = kg + kwg;

      const int w_id = idm % output_w;
      const int h_id = idm / output_w;

      // Im2col indices
      const int kernel_2d_index = idk % (kernel_h * kernel_w);
      const int kw_id = (kernel_flip)
                      ? kernel_w - kernel_2d_index % kernel_w - 1
                      : kernel_2d_index % kernel_w;
      const int kh_id = (kernel_flip)
                      ? kernel_h - kernel_2d_index / kernel_w - 1
                      : kernel_2d_index / kernel_w;
      const int c_id = idk / (kernel_h * kernel_w);
      const int h_index = -pad_h + kh_id * dilation_h + stride_h * h_id;
      const int w_index = -pad_w + kw_id * dilation_w + stride_w * w_id;

      // Loads the data from global memory into the local memory
      if (h_index >= 0 && h_index < input_h &&
          w_index >= 0 && w_index < input_w) {
        const int image_index = w_index + input_w * (h_index + input_h * c_id);
        const real result = imagegm[image_index + image_offset_batch];
        alm[kg*(WGD + PADA) + mg] = result;
      }
      else {
        SetToZero(alm[kg*(WGD + PADA) + mg]);
      }
    }
  }
}

#endif  // defined(ROUTINE_CONVGEMM) && !defined(CONVGEMM_WITH_IM2COL)

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
