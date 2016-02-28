
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xher class (see the header for information about the class).
//
// =================================================================================================

#include "internal/routines/level2/xher.h"

#include <string>

namespace clblast {
// =================================================================================================

// Specific implementations to get the memory-type based on a template argument
template <> const Precision Xher<float, float>::precision_ = Precision::kSingle;
template <> const Precision Xher<double, double>::precision_ = Precision::kDouble;
template <> const Precision Xher<float2, float>::precision_ = Precision::kComplexSingle;
template <> const Precision Xher<double2, double>::precision_ = Precision::kComplexDouble;

// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T, typename U>
Xher<T,U>::Xher(Queue &queue, Event &event, const std::string &name):
    Routine<T>(queue, event, name, {"Xger"}, precision_) {
  source_string_ =
    #include "../../kernels/level2/level2.opencl"
    #include "../../kernels/level2/xher.opencl"
  ;
}

// =================================================================================================

// Specializations to compute alpha of type 'T'
template <> float2 Xher<float2,float>::GetAlpha(const float alpha) { return float2{alpha, 0.0f}; }
template <> double2 Xher<double2,double>::GetAlpha(const double alpha) { return double2{alpha, 0.0}; }
template <> float Xher<float,float>::GetAlpha(const float alpha) { return alpha; }
template <> double Xher<double,double>::GetAlpha(const double alpha) { return alpha; }

// =================================================================================================

// The main routine
template <typename T, typename U>
StatusCode Xher<T,U>::DoHer(const Layout layout, const Triangle triangle,
                            const size_t n,
                            const U alpha,
                            const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                            const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                            const bool packed) {

  // Makes sure the dimensions are larger than zero
  if (n == 0) { return StatusCode::kInvalidDimension; }

  // The data is either in the upper or lower triangle
  const auto is_upper = ((triangle == Triangle::kUpper && layout != Layout::kRowMajor) ||
                         (triangle == Triangle::kLower && layout == Layout::kRowMajor));
  const auto is_rowmajor = (layout == Layout::kRowMajor);

  // Creates a matching version of alpha
  const auto matching_alpha = GetAlpha(alpha);

  // Tests the matrix and the vectors for validity
  auto status = StatusCode::kSuccess;
  if (packed) { status = TestMatrixAP(n, a_buffer, a_offset, sizeof(T)); }
  else { status = TestMatrixA(n, n, a_buffer, a_offset, a_ld, sizeof(T)); }
  if (ErrorIn(status)) { return status; }
  status = TestVectorX(n, x_buffer, x_offset, x_inc, sizeof(T));
  if (ErrorIn(status)) { return status; }

  // If alpha is zero an update is not required
  if (alpha == U{0}) { return StatusCode::kSuccess; }

  // Retrieves the Xgemv kernel from the compiled binary
  try {
    auto& program = GetProgramFromCache();
    auto kernel = Kernel(program, "Xher");

    // Sets the kernel arguments
    kernel.SetArgument(0, static_cast<int>(n));
    kernel.SetArgument(1, matching_alpha);
    kernel.SetArgument(2, x_buffer());
    kernel.SetArgument(3, static_cast<int>(x_offset));
    kernel.SetArgument(4, static_cast<int>(x_inc));
    kernel.SetArgument(5, a_buffer());
    kernel.SetArgument(6, static_cast<int>(a_offset));
    kernel.SetArgument(7, static_cast<int>(a_ld));
    kernel.SetArgument(8, static_cast<int>(is_upper));
    kernel.SetArgument(9, static_cast<int>(is_rowmajor));

    // Launches the kernel
    auto global_one = CeilDiv(Ceil(n, db_["WGS1"]), db_["WPT"]);
    auto global_two = CeilDiv(Ceil(n, db_["WGS2"]), db_["WPT"]);
    auto global = std::vector<size_t>{global_one, global_two};
    auto local = std::vector<size_t>{db_["WGS1"], db_["WGS2"]};
    status = RunKernel(kernel, global, local);
    if (ErrorIn(status)) { return status; }

    // Waits for all kernels to finish
    queue_.Finish();

    // Succesfully finished the computation
    return StatusCode::kSuccess;
  } catch (...) { return StatusCode::kInvalidKernel; }
}

// =================================================================================================

// Compiles the templated class
template class Xher<float, float>;
template class Xher<double, double>;
template class Xher<float2, float>;
template class Xher<double2, double>;

// =================================================================================================
} // namespace clblast
