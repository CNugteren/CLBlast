
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xgbmv class (see the header for information about the class).
//
// =================================================================================================

#include "internal/routines/level2/xgbmv.h"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xgbmv<T>::Xgbmv(Queue &queue, Event &event, const std::string &name):
    Xgemv<T>(queue, event, name) {
}

// =================================================================================================

// The main routine
template <typename T>
StatusCode Xgbmv<T>::DoGbmv(const Layout layout, const Transpose a_transpose,
                            const size_t m, const size_t n, const size_t kl, const size_t ku,
                            const T alpha,
                            const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                            const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                            const T beta,
                            const Buffer<T> &y_buffer, const size_t y_offset, const size_t y_inc) {

  // Makes sure all dimensions are larger than zero
  if (n == 0 || m == 0) { return StatusCode::kInvalidDimension; }

  //
  auto rotated = (layout == Layout::kRowMajor);
  auto t_one = (rotated) ? n : m;
  auto t_two = (rotated) ? m : n;
  auto a_one = kl+ku+1;
  auto a_two = (rotated) ? m : n;

  // Checks for validity of the A matrix
  auto status = StatusCode::kSuccess;
  if (a_ld < a_one) { return StatusCode::kInvalidLeadDimA; }
  try {
    auto required_size = (a_ld*a_two + a_offset)*sizeof(T);
    auto buffer_size = a_buffer.GetSize();
    if (buffer_size < required_size) { return StatusCode::kInsufficientMemoryA; }
  } catch (...) { return StatusCode::kInvalidMatrixA; }

  // Temporary buffer to generalize the input matrix
  try {
    auto t_buffer = Buffer<T>(context_, t_one*t_two);

    // Creates a general matrix from the input to be able to run the regular Xgemv routine
    try {
      auto& program = GetProgramFromCache();
      auto kernel = Kernel(program, "GeneralBandedToGeneral");

      // Sets the arguments for the matrix transform kernel
      kernel.SetArgument(0, static_cast<int>(a_one));
      kernel.SetArgument(1, static_cast<int>(a_two));
      kernel.SetArgument(2, static_cast<int>(a_ld));
      kernel.SetArgument(3, static_cast<int>(a_offset));
      kernel.SetArgument(4, a_buffer());
      kernel.SetArgument(5, static_cast<int>(t_one));
      kernel.SetArgument(6, static_cast<int>(t_two));
      kernel.SetArgument(7, static_cast<int>(t_one));
      kernel.SetArgument(8, static_cast<int>(0));
      kernel.SetArgument(9, t_buffer());
      kernel.SetArgument(10, static_cast<int>(layout));
      if (rotated) {
        kernel.SetArgument(11, static_cast<int>(ku));
        kernel.SetArgument(12, static_cast<int>(kl));
      }
      else {
        kernel.SetArgument(11, static_cast<int>(kl));
        kernel.SetArgument(12, static_cast<int>(ku));
      }

      // Uses the common matrix-transforms thread configuration
      auto global = std::vector<size_t>{Ceil(CeilDiv(t_one, db_["PAD_WPTX"]), db_["PAD_DIMX"]),
                                        Ceil(CeilDiv(t_two, db_["PAD_WPTY"]), db_["PAD_DIMY"])};
      auto local = std::vector<size_t>{db_["PAD_DIMX"], db_["PAD_DIMY"]};
      status = RunKernel(kernel, global, local);
      if (ErrorIn(status)) { return status; }

      // Runs the regular Xgemv code
      status = DoGemv(layout, a_transpose, m, n, alpha,
                      t_buffer, 0, t_one,
                      x_buffer, x_offset, x_inc, beta,
                      y_buffer, y_offset, y_inc);

      // Return the status of the Xgemv routine
      return status;
    } catch (...) { return StatusCode::kInvalidKernel; }
  } catch (...) { return StatusCode::kTempBufferAllocFailure; }
}

// =================================================================================================

// Compiles the templated class
template class Xgbmv<float>;
template class Xgbmv<double>;
template class Xgbmv<float2>;
template class Xgbmv<double2>;

// =================================================================================================
} // namespace clblast
