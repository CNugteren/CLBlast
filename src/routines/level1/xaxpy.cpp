
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xaxpy class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level1/xaxpy.hpp"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xaxpy<T>::Xaxpy(Queue &queue, EventPointer event, const std::string &name):
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>()) {
  source_string_ =
    #include "../../kernels/level1/level1.opencl"
    #include "../../kernels/level1/xaxpy.opencl"
  ;
}

// =================================================================================================

// The main routine
template <typename T>
StatusCode Xaxpy<T>::DoAxpy(const size_t n, const T alpha,
                            const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                            const Buffer<T> &y_buffer, const size_t y_offset, const size_t y_inc) {

  // Makes sure all dimensions are larger than zero
  if (n == 0) { return StatusCode::kInvalidDimension; }

  // Tests the vectors for validity
  auto status = TestVectorX(n, x_buffer, x_offset, x_inc);
  if (ErrorIn(status)) { return status; }
  status = TestVectorY(n, y_buffer, y_offset, y_inc);
  if (ErrorIn(status)) { return status; }

  // Determines whether or not the fast-version can be used
  bool use_fast_kernel = (x_offset == 0) && (x_inc == 1) &&
                         (y_offset == 0) && (y_inc == 1) &&
                         IsMultiple(n, db_["WGS"]*db_["WPT"]*db_["VW"]);

  // If possible, run the fast-version of the kernel
  auto kernel_name = (use_fast_kernel) ? "XaxpyFast" : "Xaxpy";

  // Retrieves the Xaxpy kernel from the compiled binary
  try {
    const auto program = GetProgramFromCache(context_, PrecisionValue<T>(), routine_name_);
    auto kernel = Kernel(program, kernel_name);

    // Sets the kernel arguments
    if (use_fast_kernel) {
      kernel.SetArgument(0, static_cast<int>(n));
      kernel.SetArgument(1, GetRealArg(alpha));
      kernel.SetArgument(2, x_buffer());
      kernel.SetArgument(3, y_buffer());
    }
    else {
      kernel.SetArgument(0, static_cast<int>(n));
      kernel.SetArgument(1, GetRealArg(alpha));
      kernel.SetArgument(2, x_buffer());
      kernel.SetArgument(3, static_cast<int>(x_offset));
      kernel.SetArgument(4, static_cast<int>(x_inc));
      kernel.SetArgument(5, y_buffer());
      kernel.SetArgument(6, static_cast<int>(y_offset));
      kernel.SetArgument(7, static_cast<int>(y_inc));
    }

    // Launches the kernel
    if (use_fast_kernel) {
      auto global = std::vector<size_t>{CeilDiv(n, db_["WPT"]*db_["VW"])};
      auto local = std::vector<size_t>{db_["WGS"]};
      status = RunKernel(kernel, queue_, device_, global, local, event_);
    }
    else {
      auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
      auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
      auto local = std::vector<size_t>{db_["WGS"]};
      status = RunKernel(kernel, queue_, device_, global, local, event_);
    }
    if (ErrorIn(status)) { return status; }

    // Succesfully finished the computation
    return StatusCode::kSuccess;
  } catch (...) { return StatusCode::kInvalidKernel; }
}

// =================================================================================================

// Compiles the templated class
template class Xaxpy<half>;
template class Xaxpy<float>;
template class Xaxpy<double>;
template class Xaxpy<float2>;
template class Xaxpy<double2>;

// =================================================================================================
} // namespace clblast
