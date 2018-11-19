
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// This file contains the col2im kernel, taken from:
// https://gist.github.com/vbkaisetsu/a98299df827f9a5245635f646c1d94be
// Credits go to https://github.com/vbkaisetsu
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// Work-group size parameters re-used from the 'copy' kernel
#ifndef COPY_DIMX
  #define COPY_DIMX 8      // Local workgroup size in the first dimension (w)
#endif
#ifndef COPY_DIMY
  #define COPY_DIMY 8      // Local workgroup size in the second dimension (h)
#endif

// =================================================================================================

inline int grid_ceil(const int x, const int step) {
  return x > 0 ? ((x - 1) / step + 1) * step : x / step * step;
}

// Main body of the kernel
INLINE_FUNC void Xcol2im(const int input_h, const int input_w, const int channels,
                         const int output_h, const int output_w,
                         const int kernel_h, const int kernel_w,
                         const int pad_h, const int pad_w,
                         const int stride_h, const int stride_w,
                         const int dilation_h, const int dilation_w,
                         const int stride_bez_h, const int stride_bez_w,
                         const int dilation_bez_h, const int dilation_bez_w,
                         const int gcd_h, const int gcd_w,
                         const bool kernel_flip,
                         const __global real* restrict col_buffer, const int col_offset,
                         __global real* im_buffer, const int im_offset) {

  const int input_h_scaled = (input_h - 1) / gcd_h + 1;

  // Thread IDs
  const int gcd_scale_w = get_global_id(0) + (pad_w - 1) / gcd_w + 1;
  const int gcd_scale_h = ((int) get_global_id(1)) % input_h_scaled + (pad_h - 1) / gcd_h + 1;
  const int c_id = ((int) get_global_id(1)) / input_h_scaled;

  const int w_index = gcd_scale_w * gcd_w - pad_w;
  const int h_index = gcd_scale_h * gcd_h - pad_h;
  const int th_step = stride_h * dilation_h / gcd_h;
  const int th_begin = grid_ceil(max(-stride_bez_h * gcd_scale_h * stride_h,
                                     (dilation_bez_h * gcd_scale_h - kernel_h + 1) * dilation_h),
                                 th_step);
  const int th_end = min((output_h - stride_bez_h * gcd_scale_h) * stride_h,
                         (dilation_bez_h * gcd_scale_h + 1) * dilation_h);
  const int tw_step = stride_w * dilation_w / gcd_w;
  const int tw_begin = grid_ceil(max(-stride_bez_w * gcd_scale_w * stride_w,
                                     (dilation_bez_w * gcd_scale_w - kernel_w + 1) * dilation_w),
                                 tw_step);
  const int tw_end = min((output_w - stride_bez_w * gcd_scale_w) * stride_w,
                         (dilation_bez_w * gcd_scale_w + 1) * dilation_w);
  if (w_index < input_w && c_id < channels) {
    real val;
    SetToZero(val);
    for (int th = th_begin; th < th_end; th += th_step) {
      for (int tw = tw_begin; tw < tw_end; tw += tw_step) {
        const int kh_id = -th / dilation_h + dilation_bez_h * gcd_scale_h;
        const int kw_id = -tw / dilation_w + dilation_bez_w * gcd_scale_w;
        const int h_id = th / stride_h + stride_bez_h * gcd_scale_h;
        const int w_id = tw / stride_w + stride_bez_w * gcd_scale_w;
        const int kernel_index = (kernel_flip)
                               ? kernel_h * kernel_w - kw_id - kernel_w * kh_id - 1
                               : kw_id + kernel_w * kh_id;
        const int patch_index = w_id + output_w * h_id;
        const int output_index = patch_index + kernel_index * output_w * output_h +
                                 c_id * output_w * output_h * kernel_h * kernel_w;
        Add(val, val, col_buffer[output_index + col_offset]);
      }
    }

    // Accumulates the resulting value with the existing im-buffer (+= val)
    const int input_index = w_index + input_w * (h_index + input_h * c_id);
    real im_buffer_value = im_buffer[input_index + im_offset];
    Add(im_buffer[input_index + im_offset], im_buffer_value, val);
  }
}

// =================================================================================================

// Kernel flip version of the Xcol2im kernel (for convolution)
__kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))
void Xcol2imKernelFlip(const int input_h, const int input_w, const int channels,
                       const int output_h, const int output_w,
                       const int kernel_h, const int kernel_w,
                       const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       const int stride_bez_h, const int stride_bez_w,
                       const int dilation_bez_h, const int dilation_bez_w,
                       const int gcd_h, const int gcd_w,
                       const __global real* restrict col_buffer, const int col_offset,
                       __global real* im_buffer, const int im_offset) {
  const bool kernel_flip = true;
  Xcol2im(input_h, input_w, channels, output_h, output_w, kernel_h, kernel_w,
          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
          stride_bez_h, stride_bez_w, dilation_bez_h, dilation_bez_w, gcd_h, gcd_w,
          kernel_flip,
          col_buffer, col_offset, im_buffer, im_offset);
}

// Normal version of the Xcol2im kernel (for cross-correlation)
__kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))
void Xcol2imKernelNormal(const int input_h, const int input_w, const int channels,
                         const int output_h, const int output_w,
                         const int kernel_h, const int kernel_w,
                         const int pad_h, const int pad_w,
                         const int stride_h, const int stride_w,
                         const int dilation_h, const int dilation_w,
                         const int stride_bez_h, const int stride_bez_w,
                         const int dilation_bez_h, const int dilation_bez_w,
                         const int gcd_h, const int gcd_w,
                         const __global real* restrict col_buffer, const int col_offset,
                         __global real* im_buffer, const int im_offset) {
  const bool kernel_flip = false;
  Xcol2im(input_h, input_w, channels, output_h, output_w, kernel_h, kernel_w,
          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
          stride_bez_h, stride_bez_w, dilation_bez_h, dilation_bez_w, gcd_h, gcd_w,
          kernel_flip,
          col_buffer, col_offset, im_buffer, im_offset);
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
