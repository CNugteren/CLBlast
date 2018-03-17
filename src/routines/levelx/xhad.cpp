
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xhad class (see the header for information about the class).
//
// =================================================================================================

#include "routines/levelx/xhad.hpp"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xhad<T>::Xhad(Queue &queue, EventPointer event, const std::string &name):
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
#include "../../kernels/level1/level1.opencl"
#include "../../kernels/level1/xhad.opencl"
    }) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xhad<T>::DoHad(const size_t n, const T alpha,
                    const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                    const Buffer<T> &y_buffer, const size_t y_offset, const size_t y_inc, const T beta,
                    const Buffer<T> &z_buffer, const size_t z_offset, const size_t z_inc) {

  // Makes sure all dimensions are larger than zero
  if (n == 0) { throw BLASError(StatusCode::kInvalidDimension); }

  // Tests the vectors for validity
  TestVectorX(n, x_buffer, x_offset, x_inc);
  TestVectorY(n, y_buffer, y_offset, y_inc);
  TestVectorY(n, z_buffer, z_offset, z_inc); // TODO: Make a TestVectorZ function with error codes

  // Determines whether or not the fast-version can be used
  const auto use_faster_kernel = (x_offset == 0) && (x_inc == 1) &&
                                 (y_offset == 0) && (y_inc == 1) &&
                                 (z_offset == 0) && (z_inc == 1) &&
                                 IsMultiple(n, db_["WPT"]*db_["VW"]);
  const auto use_fastest_kernel = use_faster_kernel &&
                                  IsMultiple(n, db_["WGS"]*db_["WPT"]*db_["VW"]);

  // If possible, run the fast-version of the kernel
  const auto kernel_name = (use_fastest_kernel) ? "XhadFastest" :
                           (use_faster_kernel) ? "XhadFaster" : "Xhad";

  // Retrieves the Xhad kernel from the compiled binary
  auto kernel = Kernel(program_, kernel_name);

  // Sets the kernel arguments
  if (use_faster_kernel || use_fastest_kernel) {
    kernel.SetArgument(0, static_cast<int>(n));
    kernel.SetArgument(1, GetRealArg(alpha));
    kernel.SetArgument(2, GetRealArg(beta));
    kernel.SetArgument(3, x_buffer());
    kernel.SetArgument(4, y_buffer());
    kernel.SetArgument(5, z_buffer());
  }
  else {
    kernel.SetArgument(0, static_cast<int>(n));
    kernel.SetArgument(1, GetRealArg(alpha));
    kernel.SetArgument(2, GetRealArg(beta));
    kernel.SetArgument(3, x_buffer());
    kernel.SetArgument(4, static_cast<int>(x_offset));
    kernel.SetArgument(5, static_cast<int>(x_inc));
    kernel.SetArgument(6, y_buffer());
    kernel.SetArgument(7, static_cast<int>(y_offset));
    kernel.SetArgument(8, static_cast<int>(y_inc));
    kernel.SetArgument(9, z_buffer());
    kernel.SetArgument(10, static_cast<int>(z_offset));
    kernel.SetArgument(11, static_cast<int>(z_inc));
  }

  // Launches the kernel
  if (use_fastest_kernel) {
    auto global = std::vector<size_t>{CeilDiv(n, db_["WPT"]*db_["VW"])};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
  }
  else if (use_faster_kernel) {
    auto global = std::vector<size_t>{Ceil(CeilDiv(n, db_["WPT"]*db_["VW"]), db_["WGS"])};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
  }
  else {
    const auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
  }
}

// =================================================================================================

// Compiles the templated class
template class Xhad<half>;
template class Xhad<float>;
template class Xhad<double>;
template class Xhad<float2>;
template class Xhad<double2>;

// =================================================================================================
} // namespace clblast
