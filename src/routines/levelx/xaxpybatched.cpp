
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
    Xaxpy<T>(queue, event, name) {
}

// =================================================================================================

// The main routine
template <typename T>
void XaxpyBatched<T>::DoAxpyBatched(const size_t n, const std::vector<T> &alphas,
                                    const std::vector<Buffer<T>> &x_buffers, const size_t x_inc,
                                    const std::vector<Buffer<T>> &y_buffers, const size_t y_inc,
                                    const size_t batch_count) {
  if (batch_count < 1) { throw BLASError(StatusCode::kInvalidBatchCount); }
  if (alphas.size() != batch_count) { throw BLASError(StatusCode::kInvalidBatchCount); }
  if (x_buffers.size() != batch_count) { throw BLASError(StatusCode::kInvalidBatchCount); }
  if (y_buffers.size() != batch_count) { throw BLASError(StatusCode::kInvalidBatchCount); }

  // Naive implementation: calls regular Axpy multiple times
  for (auto batch = size_t{0}; batch < batch_count; ++batch) {
    DoAxpy(n, alphas[batch],
           x_buffers[batch], 0, x_inc,
           y_buffers[batch], 0, y_inc);
  }
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
