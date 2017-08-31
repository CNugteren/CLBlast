
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the im2col kernel.
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

__kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))
void im2col(const int input_h, const int input_w,
            const int output_h, const int output_w,
            const int kernel_h, const int kernel_w,
            const int pad_h, const int pad_w,
            const int stride_h, const int stride_w,
            const int dilation_h, const int dilation_w,
            const __global real* restrict im_buffer, const int im_offset,
            __global real* col_buffer, const int col_offset) {

  // Thread IDs
  const int w_id = get_global_id(0); // image width, max 'output_w'
  const int h_id = get_global_id(1) % output_h; // image height, max 'output_h'
  const int c_id = get_global_id(1) / output_h; // input channels
  if (h_id < output_h && w_id < output_w) {

    #pragma unroll
    for (int kh_id = 0; kh_id < kernel_h; ++kh_id) { // kernel height
      #pragma unroll
      for (int kw_id = 0; kw_id < kernel_w; ++kw_id) { // kernel width

        // Retrieves the input value
        const int h_index = -pad_h + kh_id * dilation_h + stride_h * h_id;
        const int w_index = -pad_w + kw_id * dilation_w + stride_w * w_id;
        real val;
        if (h_index >= 0 && h_index < input_h &&
            w_index >= 0 && w_index < input_w) {
          const int input_index = w_index + input_w * (h_index + input_h * c_id);
          val = im_buffer[input_index + im_offset];
        }
        else {
          SetToZero(val);
        }

        // Sets the output value
        const int kernel_index = kw_id + kernel_w * kh_id;
        const int patch_index = w_id + output_w * h_id;
        const int output_index = patch_index + kernel_index * output_w * output_h +
                                  c_id * output_w * output_h * kernel_h * kernel_w;
        col_buffer[output_index + col_offset] = val;
      }
    }
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
