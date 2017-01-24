
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xger class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level2/xger.hpp"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xger<T>::Xger(Queue &queue, EventPointer event, const std::string &name):
    Routine(queue, event, name, {"Xger"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level2/level2.opencl"
    #include "../../kernels/level2/xger.opencl"
    }) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xger<T>::DoGer(const Layout layout,
                    const size_t m, const size_t n,
                    const T alpha,
                    const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                    const Buffer<T> &y_buffer, const size_t y_offset, const size_t y_inc,
                    const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld) {

  // Makes sure all dimensions are larger than zero
  if (m == 0 || n == 0) { throw BLASError(StatusCode::kInvalidDimension); }

  // Computes whether or not the matrix has an alternative layout (row or column-major).
  const auto a_is_rowmajor = (layout == Layout::kRowMajor);
  const auto a_one = (a_is_rowmajor) ? n : m;
  const auto a_two = (a_is_rowmajor) ? m : n;

  // Tests the matrix and the vectors for validity
  TestMatrixA(a_one, a_two, a_buffer, a_offset, a_ld);
  TestVectorX(m, x_buffer, x_offset, x_inc);
  TestVectorY(n, y_buffer, y_offset, y_inc);

  // Retrieves the kernel from the compiled binary
  auto kernel = Kernel(program_, "Xger");

  // Sets the kernel arguments
  kernel.SetArgument(0, static_cast<int>(a_one));
  kernel.SetArgument(1, static_cast<int>(a_two));
  kernel.SetArgument(2, GetRealArg(alpha));
  kernel.SetArgument(3, x_buffer());
  kernel.SetArgument(4, static_cast<int>(x_offset));
  kernel.SetArgument(5, static_cast<int>(x_inc));
  kernel.SetArgument(6, y_buffer());
  kernel.SetArgument(7, static_cast<int>(y_offset));
  kernel.SetArgument(8, static_cast<int>(y_inc));
  kernel.SetArgument(9, a_buffer());
  kernel.SetArgument(10, static_cast<int>(a_offset));
  kernel.SetArgument(11, static_cast<int>(a_ld));
  kernel.SetArgument(12, static_cast<int>(a_is_rowmajor));

  // Launches the kernel
  auto a_one_ceiled = Ceil(CeilDiv(a_one, db_["WPT"]), db_["WGS1"]);
  auto a_two_ceiled = Ceil(CeilDiv(a_two, db_["WPT"]), db_["WGS2"]);
  auto global = std::vector<size_t>{a_one_ceiled, a_two_ceiled};
  auto local = std::vector<size_t>{db_["WGS1"], db_["WGS2"]};
  RunKernel(kernel, queue_, device_, global, local, event_);
}

// =================================================================================================

// Compiles the templated class
template class Xger<half>;
template class Xger<float>;
template class Xger<double>;
template class Xger<float2>;
template class Xger<double2>;

// =================================================================================================
} // namespace clblast
