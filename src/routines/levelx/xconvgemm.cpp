
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
    Routine(queue, event, name, {"XgemmDirect"},
        PrecisionValue<T>(), {}, {
            #include "../../kernels/level3/level3.opencl"
            , // separated in multiple parts to prevent C1091 in MSVC 2013
            #include "../../kernels/level3/xgemm_direct_part1.opencl"
            #include "../../kernels/level3/xgemm_direct_part2.opencl"
            #include "../../kernels/level3/xgemm_direct_part3.opencl"
            , // separated in multiple parts to prevent C1091 in MSVC 2013
            #include "../../kernels/level3/xgemm_direct_batched.opencl"
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

  // Tests for a valid batch count
  if (batch_count == 0) {
    throw BLASError(StatusCode::kInvalidBatchCount);
  }

  // Makes sure all dimensions are larger than zero
  if ((channels == 0) || (height == 0) || (width == 0) || (num_kernels == 0)) {
    throw BLASError(StatusCode::kInvalidDimension);
  }

  // Sets the output height and width
  const auto size_h = height + 2 * pad_h;
  const auto padding_h = dilation_h * (kernel_h - 1) + 1;
  const auto output_h = (size_h >= padding_h) ? (size_h - padding_h) / stride_h + 1 : 1;
  const auto size_w = width + 2 * pad_w;
  const auto padding_w = dilation_w * (kernel_w - 1) + 1;
  const auto output_w = (size_w >= padding_w) ? (size_w - padding_w) / stride_w + 1 : 1;

  // Sets other useful variables
  const auto patch_size = kernel_h * kernel_w * channels;
  const auto num_patches = output_h * output_w;

  // Approach: im2col + GEMM
  //      result = GEMM(im2col(image), kernel)

  // Temporary col matrix
  const auto col_size = patch_size * num_patches * batch_count;
  auto col_buffer = Buffer<T>(context_, col_size);

  // Loops over each batch
  for (auto batch_id = size_t{0}; batch_id < batch_count; ++batch_id) {

    // im2col
    const auto im_batch_offset = batch_id * channels * height * width + im_offset;
    const auto col_batch_offset = batch_id * patch_size * num_patches;
    auto im2col_event = Event();
    auto im2col = Xim2col<T>(queue_, im2col_event.pointer());
    im2col.DoIm2col(channels, height, width, kernel_h, kernel_w,
                    pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                    im_buffer, im_batch_offset,
                    col_buffer, col_batch_offset);
    im2col_event.WaitForCompletion();
  }

  // Strided batched GEMM: C (result) = alpha (1) * A (col) * B (kernel) + beta (0) * C (result)
  const auto m = num_patches;
  const auto n = num_kernels;
  const auto k = patch_size;
  const auto col_ld = m;
  const auto kernel_ld = k;
  const auto result_ld = m;
  const auto col_stride = patch_size * num_patches;
  const auto kernel_stride = size_t{0}; // applies the same kernel to all batches
  const auto result_stride = num_kernels * output_h * output_w;

  // Computes the transpose/conjugate options and sets the a/b/c sizes based on that
  bool col_do_transpose, kernel_do_transpose, result_do_transpose, col_conjugate, kernel_conjugate;
  size_t col_one, col_two, kernel_one, kernel_two, result_one, result_two;
  Xgemm<T>::ProcessArguments(Layout::kColMajor, Transpose::kNo, Transpose::kNo, m, n, k,
                             col_one, col_two, kernel_one, kernel_two, result_one, result_two,
                             col_do_transpose, kernel_do_transpose,
                             result_do_transpose, col_conjugate, kernel_conjugate, 0);

  // Tests the matrices for validity
  for (auto batch = size_t{0}; batch < batch_count; ++batch) {
    TestMatrixA(col_one, col_two, col_buffer, col_stride * batch, col_ld);
    TestMatrixB(kernel_one, kernel_two, kernel_buffer, kernel_offset + kernel_stride * batch, kernel_ld);
    TestMatrixC(result_one, result_two, result_buffer, result_offset + result_stride * batch, result_ld);
  }

  // Retrieves the proper XgemmDirect kernel from the compiled binary
  const auto name = (col_do_transpose) ? (kernel_do_transpose ? "XgemmDirectStridedBatchedTT" : "XgemmDirectStridedBatchedTN") :
                    (kernel_do_transpose ? "XgemmDirectStridedBatchedNT" : "XgemmDirectStridedBatchedNN");
  auto kernel = Kernel(program_, name);

  // Sets the kernel arguments
  kernel.SetArgument(0, static_cast<int>(m));
  kernel.SetArgument(1, static_cast<int>(n));
  kernel.SetArgument(2, static_cast<int>(k));
  kernel.SetArgument(3, GetRealArg(ConstantOne<T>()));
  kernel.SetArgument(4, GetRealArg(ConstantZero<T>()));
  kernel.SetArgument(5, col_buffer());
  kernel.SetArgument(6, static_cast<int>(0));
  kernel.SetArgument(7, static_cast<int>(col_ld));
  kernel.SetArgument(8, static_cast<int>(col_stride));
  kernel.SetArgument(9, kernel_buffer());
  kernel.SetArgument(10, static_cast<int>(kernel_offset));
  kernel.SetArgument(11, static_cast<int>(kernel_ld));
  kernel.SetArgument(12, static_cast<int>(kernel_stride));
  kernel.SetArgument(13, result_buffer());
  kernel.SetArgument(14, static_cast<int>(result_offset));
  kernel.SetArgument(15, static_cast<int>(result_ld));
  kernel.SetArgument(16, static_cast<int>(result_stride));
  kernel.SetArgument(17, static_cast<int>(result_do_transpose));
  kernel.SetArgument(18, static_cast<int>(false));
  kernel.SetArgument(19, static_cast<int>(false));

  // Computes the global and local thread sizes
  const auto m_ceiled = Ceil(m, db_["WGD"]);
  const auto n_ceiled = Ceil(n, db_["WGD"]);
  const auto global = std::vector<size_t>{
      (m_ceiled * db_["MDIMCD"]) / db_["WGD"],
      (n_ceiled * db_["NDIMCD"]) / db_["WGD"],
      batch_count
  };
  const auto local = std::vector<size_t>{db_["MDIMCD"], db_["NDIMCD"], 1};

  // Launches the kernel
  RunKernel(kernel, queue_, device_, global, local, event_);
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
