
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xher2 class (see the header for information about the class).
//
// =================================================================================================

#include "internal/routines/level2/xher2.h"

#include <string>

namespace clblast {
// =================================================================================================

// Specific implementations to get the memory-type based on a template argument
template <> const Precision Xher2<half>::precision_ = Precision::kHalf;
template <> const Precision Xher2<float>::precision_ = Precision::kSingle;
template <> const Precision Xher2<double>::precision_ = Precision::kDouble;
template <> const Precision Xher2<float2>::precision_ = Precision::kComplexSingle;
template <> const Precision Xher2<double2>::precision_ = Precision::kComplexDouble;

// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xher2<T>::Xher2(Queue &queue, EventPointer event, const std::string &name):
    Routine<T>(queue, event, name, {"Xger"}, precision_) {
  source_string_ =
    #include "../../kernels/level2/level2.opencl"
    #include "../../kernels/level2/xher2.opencl"
  ;
}

// =================================================================================================

// The main routine
template <typename T>
StatusCode Xher2<T>::DoHer2(const Layout layout, const Triangle triangle,
                            const size_t n,
                            const T alpha,
                            const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                            const Buffer<T> &y_buffer, const size_t y_offset, const size_t y_inc,
                            const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                            const bool packed) {

  // Makes sure the dimensions are larger than zero
  if (n == 0) { return StatusCode::kInvalidDimension; }

  // The data is either in the upper or lower triangle
  const auto is_upper = ((triangle == Triangle::kUpper && layout != Layout::kRowMajor) ||
                         (triangle == Triangle::kLower && layout == Layout::kRowMajor));
  const auto is_rowmajor = (layout == Layout::kRowMajor);

  // Tests the matrix and the vectors for validity
  auto status = StatusCode::kSuccess;
  if (packed) { status = TestMatrixAP(n, a_buffer, a_offset); }
  else { status = TestMatrixA(n, n, a_buffer, a_offset, a_ld); }
  if (ErrorIn(status)) { return status; }
  status = TestVectorX(n, x_buffer, x_offset, x_inc);
  if (ErrorIn(status)) { return status; }
  status = TestVectorY(n, y_buffer, y_offset, y_inc);
  if (ErrorIn(status)) { return status; }

  // Upload the scalar argument as a constant buffer to the device (needed for half-precision)
  auto alpha_buffer = Buffer<T>(context_, 1);
  alpha_buffer.Write(queue_, 1, &alpha);

  // Retrieves the kernel from the compiled binary
  try {
    const auto program = GetProgramFromCache();
    auto kernel = Kernel(program, "Xher2");

    // Sets the kernel arguments
    kernel.SetArgument(0, static_cast<int>(n));
    kernel.SetArgument(1, alpha_buffer());
    kernel.SetArgument(2, x_buffer());
    kernel.SetArgument(3, static_cast<int>(x_offset));
    kernel.SetArgument(4, static_cast<int>(x_inc));
    kernel.SetArgument(5, y_buffer());
    kernel.SetArgument(6, static_cast<int>(y_offset));
    kernel.SetArgument(7, static_cast<int>(y_inc));
    kernel.SetArgument(8, a_buffer());
    kernel.SetArgument(9, static_cast<int>(a_offset));
    kernel.SetArgument(10, static_cast<int>(a_ld));
    kernel.SetArgument(11, static_cast<int>(is_upper));
    kernel.SetArgument(12, static_cast<int>(is_rowmajor));

    // Launches the kernel
    auto global_one = Ceil(CeilDiv(n, db_["WPT"]), db_["WGS1"]);
    auto global_two = Ceil(CeilDiv(n, db_["WPT"]), db_["WGS2"]);
    auto global = std::vector<size_t>{global_one, global_two};
    auto local = std::vector<size_t>{db_["WGS1"], db_["WGS2"]};
    status = RunKernel(kernel, queue_, device_, global, local, event_);
    if (ErrorIn(status)) { return status; }

    // Succesfully finished the computation
    return StatusCode::kSuccess;
  } catch (...) { return StatusCode::kInvalidKernel; }
}

// =================================================================================================

// Compiles the templated class
template class Xher2<half>;
template class Xher2<float>;
template class Xher2<double>;
template class Xher2<float2>;
template class Xher2<double2>;

// =================================================================================================
} // namespace clblast
