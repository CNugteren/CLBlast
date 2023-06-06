
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the XaxpyBatched class (see the header for information about the class).
//
// =================================================================================================

#include "routines/levelx/xaxpybatched.hpp"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
XaxpyBatched<T>::XaxpyBatched(Queue &queue, EventPointer event, const std::string &name):
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level1/level1.opencl"
    #include "../../kernels/level1/xaxpy.opencl"
    }) {
}

// =================================================================================================

// The main routine
template <typename T>
void XaxpyBatched<T>::DoAxpyBatched(const size_t n, const std::vector<T> &alphas,
                                    const Buffer<T> &x_buffer, const std::vector<size_t> &x_offsets, const size_t x_inc,
                                    const Buffer<T> &y_buffer, const std::vector<size_t> &y_offsets, const size_t y_inc,
                                    const size_t batch_count) {

  // Tests for a valid batch count
  if ((batch_count < 1) || (alphas.size() != batch_count) ||
      (x_offsets.size() != batch_count) || (y_offsets.size() != batch_count)) {
    throw BLASError(StatusCode::kInvalidBatchCount);
  }

  // Makes sure all dimensions are larger than zero
  if (n == 0) { throw BLASError(StatusCode::kInvalidDimension); }

  // Tests the vectors for validity
  for (auto batch = size_t{0}; batch < batch_count; ++batch) {
    TestVectorX(n, x_buffer, x_offsets[batch], x_inc);
    TestVectorY(n, y_buffer, y_offsets[batch], y_inc);
  }

  // Upload the arguments to the device
  auto x_offsets_int = std::vector<int>(batch_count);
  auto y_offsets_int = std::vector<int>(batch_count);
  for (auto batch = size_t{ 0 }; batch < batch_count; ++batch) {
    x_offsets_int[batch] = static_cast<int>(x_offsets[batch]);
    y_offsets_int[batch] = static_cast<int>(y_offsets[batch]);
  }
  auto x_offsets_device = Buffer<int>(context_, BufferAccess::kReadWrite, batch_count);
  auto y_offsets_device = Buffer<int>(context_, BufferAccess::kReadWrite, batch_count);
  auto alphas_device = Buffer<T>(context_, BufferAccess::kReadWrite, batch_count);
  x_offsets_device.Write(queue_, batch_count, x_offsets_int);
  y_offsets_device.Write(queue_, batch_count, y_offsets_int);
  alphas_device.Write(queue_, batch_count, alphas);

  // Retrieves the Xaxpy kernel from the compiled binary
  auto kernel = Kernel(program_, "XaxpyBatched");

  // Sets the kernel arguments
  kernel.SetArgument(0, static_cast<int>(n));
  kernel.SetArgument(1, alphas_device());
  kernel.SetArgument(2, x_buffer());
  kernel.SetArgument(3, x_offsets_device());
  kernel.SetArgument(4, static_cast<int>(x_inc));
  kernel.SetArgument(5, y_buffer());
  kernel.SetArgument(6, y_offsets_device());
  kernel.SetArgument(7, static_cast<int>(y_inc));

  // Launches the kernel
  auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
  auto global = std::vector<size_t>{n_ceiled/db_["WPT"], batch_count};
  auto local = std::vector<size_t>{db_["WGS"], 1};
  RunKernel(kernel, queue_, device_, global, local, event_);
}

// =================================================================================================

// Compiles the templated class
template class XaxpyBatched<half>;
template class XaxpyBatched<float>;
template class XaxpyBatched<double>;
template class XaxpyBatched<float2>;
template class XaxpyBatched<double2>;

// =================================================================================================
} // namespace clblast
