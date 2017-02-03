
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

#include "routines/level2/xher2.hpp"

#include <string>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xher2<T>::Xher2(Queue &queue, EventPointer event, const std::string &name):
    Routine(queue, event, name, {"Xger"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level2/level2.opencl"
    #include "../../kernels/level2/xher2.opencl"
    }) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xher2<T>::DoHer2(const Layout layout, const Triangle triangle,
                      const size_t n,
                      const T alpha,
                      const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                      const Buffer<T> &y_buffer, const size_t y_offset, const size_t y_inc,
                      const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                      const bool packed) {

  // Makes sure the dimensions are larger than zero
  if (n == 0) { throw BLASError(StatusCode::kInvalidDimension); }

  // The data is either in the upper or lower triangle
  const auto is_upper = ((triangle == Triangle::kUpper && layout != Layout::kRowMajor) ||
                         (triangle == Triangle::kLower && layout == Layout::kRowMajor));
  const auto is_rowmajor = (layout == Layout::kRowMajor);

  // Tests the matrix and the vectors for validity
  if (packed) { TestMatrixAP(n, a_buffer, a_offset); }
  else { TestMatrixA(n, n, a_buffer, a_offset, a_ld); }
  TestVectorX(n, x_buffer, x_offset, x_inc);
  TestVectorY(n, y_buffer, y_offset, y_inc);

  // Retrieves the kernel from the compiled binary
  auto kernel = Kernel(program_, "Xher2");

  // Sets the kernel arguments
  kernel.SetArgument(0, static_cast<int>(n));
  kernel.SetArgument(1, GetRealArg(alpha));
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
  RunKernel(kernel, queue_, device_, global, local, event_);
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
