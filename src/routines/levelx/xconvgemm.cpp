
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xconvgemm class (see the header for information about the class).
//
// =================================================================================================

#include "routines/levelx/xconvgemm.hpp"
#include "routines/levelx/xim2col.hpp"
#include "routines/level3/xgemm.hpp"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xconvgemm<T>::Xconvgemm(Queue &queue, EventPointer event, const std::string &name):
    Routine(queue, event, name, {"Copy"}, PrecisionValue<T>(), {}, {
#include "../../kernels/levelx/im2col.opencl"
    }) {
}

// =================================================================================================

template <typename T>
void Xconvgemm<T>::DoConvgemm(const size_t channels, const size_t height, const size_t width,
                              const size_t kernel_h, const size_t kernel_w, const size_t pad_h,
                              const size_t pad_w, const size_t stride_h, const size_t stride_w,
                              const size_t dilation_h, const size_t dilation_w,
                              const size_t num_kernels, const size_t batch_count,
                              const Buffer<T> &im_buffer, const size_t im_offset,
                              const Buffer<T> &kernel_buffer, const size_t kernel_offset,
                              const Buffer<T> &result_buffer, const size_t result_offset) {

  // Makes sure all dimensions are larger than zero
  if ((channels == 0) || (height == 0) || (width == 0) || (num_kernels == 0) || (batch_count == 0)) {
    throw BLASError(StatusCode::kInvalidDimension);
  }

  // Sets the output height and width
  const auto size_h = height + 2 * pad_h;
  const auto padding_h = dilation_h * (kernel_h - 1) + 1;
  const auto output_h = (size_h >= padding_h) ? (size_h - padding_h) / stride_h + 1 : 1;
  const auto size_w = width + 2 * pad_w;
  const auto padding_w = dilation_w * (kernel_w - 1) + 1;
  const auto output_w = (size_w >= padding_w) ? (size_w - padding_w) / stride_w + 1 : 1;

  // Temporary col matrix
  const auto patch_size = kernel_h * kernel_w * channels;
  const auto num_patches = output_h * output_w;
  const auto col_size = patch_size * num_patches;
  auto col_buffer = Buffer<T>(context_, col_size);

  // Approach: im2col + GEMM
  //      result = GEMM(im2col(image), kernel)
  for (auto batch_id = size_t{0}; batch_id < batch_count; ++batch_id) {

    // im2col
    const auto im_batch_offset = batch_id * channels * height * width + im_offset;
    auto im2col_event = Event();
    auto im2col = Xim2col<T>(queue_, im2col_event.pointer());
    im2col.DoIm2col(channels, height, width, kernel_h, kernel_w,
                    pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                    im_buffer, im_batch_offset,
                    col_buffer, 0);
    im2col_event.WaitForCompletion();

    // GEMM: C (result) = alpha (1) * A (col) * B (kernel) + beta (0) * C (result)
    const auto m = num_patches;
    const auto n = num_kernels;
    const auto k = patch_size;
    const auto col_gemm_offset = size_t{0}; // A
    const auto kernel_gemm_offset = kernel_offset; // B
    const auto result_gemm_offset = batch_id * num_kernels * output_h * output_w + result_offset; // C
    const auto col_ld = m;
    const auto kernel_ld = k;
    const auto result_ld = m;
    auto gemm_event = Event();
    auto gemm = Xgemm<T>(queue_, gemm_event.pointer());
    gemm.DoGemm(Layout::kColMajor, Transpose::kNo, Transpose::kNo,
                m, n, k, ConstantOne<T>(),
                col_buffer, col_gemm_offset, col_ld,
                kernel_buffer, kernel_gemm_offset, kernel_ld, ConstantZero<T>(),
                result_buffer, result_gemm_offset, result_ld);
    gemm_event.WaitForCompletion();
  }
}

// =================================================================================================

// Compiles the templated class
template class Xconvgemm<half>;
template class Xconvgemm<float>;
template class Xconvgemm<double>;
template class Xconvgemm<float2>;
template class Xconvgemm<double2>;

// =================================================================================================
} // namespace clblast
