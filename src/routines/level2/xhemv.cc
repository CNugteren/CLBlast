
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xhemv class (see the header for information about the class).
//
// =================================================================================================

#include "internal/routines/level2/xhemv.h"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xhemv<T>::Xhemv(Queue &queue, Event &event, const std::string &name):
    Xgemv<T>(queue, event, name) {
}

// =================================================================================================

// The main routine
template <typename T>
StatusCode Xhemv<T>::DoHemv(const Layout layout, const Triangle triangle,
                            const size_t n,
                            const T alpha,
                            const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                            const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                            const T beta,
                            const Buffer<T> &y_buffer, const size_t y_offset, const size_t y_inc) {

  // Makes sure all dimensions are larger than zero
  if (n == 0) { return StatusCode::kInvalidDimension; }

  // Checks for validity of the squared A matrix
  auto status = TestMatrixA(n, n, a_buffer, a_offset, a_ld, sizeof(T));
  if (ErrorIn(status)) { return status; }

  // Determines which kernel to run based on the layout (the Xgemv kernel assumes column-major as
  // default) and on whether we are dealing with an upper or lower triangle of the hermitian matrix
  bool is_upper = ((triangle == Triangle::kUpper && layout != Layout::kRowMajor) ||
                   (triangle == Triangle::kLower && layout == Layout::kRowMajor));
  auto kernel_name = (is_upper) ? "HermUpperToSquared" : "HermLowerToSquared";

  // Temporary buffer for a copy of the hermitian matrix
  try {
    auto temp_herm = Buffer<T>(context_, n*n);

    // Creates a general matrix from the hermitian matrix to be able to run the regular Xgemv
    // routine afterwards
    try {
      auto& program = GetProgramFromCache();
      auto kernel = Kernel(program, kernel_name);

      // Sets the arguments for the hermitian-to-squared kernel
      kernel.SetArgument(0, static_cast<int>(n));
      kernel.SetArgument(1, static_cast<int>(a_ld));
      kernel.SetArgument(2, static_cast<int>(a_offset));
      kernel.SetArgument(3, a_buffer());
      kernel.SetArgument(4, static_cast<int>(n));
      kernel.SetArgument(5, static_cast<int>(n));
      kernel.SetArgument(6, static_cast<int>(0));
      kernel.SetArgument(7, temp_herm());

      // Uses the common padding kernel's thread configuration. This is allowed, since the
      // hermitian-to-squared kernel uses the same parameters.
      auto global = std::vector<size_t>{Ceil(CeilDiv(n, db_["PAD_WPTX"]), db_["PAD_DIMX"]),
                                        Ceil(CeilDiv(n, db_["PAD_WPTY"]), db_["PAD_DIMY"])};
      auto local = std::vector<size_t>{db_["PAD_DIMX"], db_["PAD_DIMY"]};
      status = RunKernel(kernel, global, local);
      if (ErrorIn(status)) { return status; }

      // Runs the regular Xgemv code
      status = DoGemv(layout, Transpose::kNo, n, n, alpha,
                      temp_herm, 0, n,
                      x_buffer, x_offset, x_inc, beta,
                      y_buffer, y_offset, y_inc);

      // Return the status of the Xgemv routine
      return status;
    } catch (...) { return StatusCode::kInvalidKernel; }
  } catch (...) { return StatusCode::kTempBufferAllocFailure; }
}

// =================================================================================================

// Compiles the templated class
template class Xhemv<float2>;
template class Xhemv<double2>;

// =================================================================================================
} // namespace clblast
