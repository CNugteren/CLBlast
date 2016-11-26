
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xscal class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level1/xscal.hpp"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xscal<T>::Xscal(Queue &queue, EventPointer event, const std::string &name):
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level1/level1.opencl"
    #include "../../kernels/level1/xscal.opencl"
    }) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xscal<T>::DoScal(const size_t n, const T alpha,
                      const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc) {

  // Makes sure all dimensions are larger than zero
  if (n == 0) { throw BLASError(StatusCode::kInvalidDimension); }

  // Tests the vector for validity
  TestVectorX(n, x_buffer, x_offset, x_inc);

  // Determines whether or not the fast-version can be used
  bool use_fast_kernel = (x_offset == 0) && (x_inc == 1) &&
                         IsMultiple(n, db_["WGS"]*db_["WPT"]*db_["VW"]);

  // If possible, run the fast-version of the kernel
  auto kernel_name = (use_fast_kernel) ? "XscalFast" : "Xscal";

  // Retrieves the Xscal kernel from the compiled binary
  auto kernel = Kernel(program_, kernel_name);

  // Sets the kernel arguments
  if (use_fast_kernel) {
    kernel.SetArgument(0, static_cast<int>(n));
    kernel.SetArgument(1, GetRealArg(alpha));
    kernel.SetArgument(2, x_buffer());
  }
  else {
    kernel.SetArgument(0, static_cast<int>(n));
    kernel.SetArgument(1, GetRealArg(alpha));
    kernel.SetArgument(2, x_buffer());
    kernel.SetArgument(3, static_cast<int>(x_offset));
    kernel.SetArgument(4, static_cast<int>(x_inc));
  }

  // Launches the kernel
  if (use_fast_kernel) {
    auto global = std::vector<size_t>{CeilDiv(n, db_["WPT"]*db_["VW"])};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
  }
  else {
    auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
  }
}

// =================================================================================================

// Compiles the templated class
template class Xscal<half>;
template class Xscal<float>;
template class Xscal<double>;
template class Xscal<float2>;
template class Xscal<double2>;

// =================================================================================================
} // namespace clblast
