
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xdot class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level1/xdot.hpp"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xdot<T>::Xdot(Queue &queue, EventPointer event, const std::string &name):
    Routine(queue, event, name, {"Xdot"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level1/xdot.opencl"
    }) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xdot<T>::DoDot(const size_t n,
                    const Buffer<T> &dot_buffer, const size_t dot_offset,
                    const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                    const Buffer<T> &y_buffer, const size_t y_offset, const size_t y_inc,
                    const bool do_conjugate) {

  // Makes sure all dimensions are larger than zero
  if (n == 0) { throw BLASError(StatusCode::kInvalidDimension); }

  // Tests the vectors for validity
  TestVectorX(n, x_buffer, x_offset, x_inc);
  TestVectorY(n, y_buffer, y_offset, y_inc);
  TestVectorScalar(1, dot_buffer, dot_offset);

  // Retrieves the Xdot kernels from the compiled binary
  auto kernel1 = Kernel(program_, "Xdot");
  auto kernel2 = Kernel(program_, "XdotEpilogue");

  // Creates the buffer for intermediate values
  auto temp_size = 2*db_["WGS2"];
  auto temp_buffer = Buffer<T>(context_, temp_size);

  // Sets the kernel arguments
  kernel1.SetArgument(0, static_cast<int>(n));
  kernel1.SetArgument(1, x_buffer());
  kernel1.SetArgument(2, static_cast<int>(x_offset));
  kernel1.SetArgument(3, static_cast<int>(x_inc));
  kernel1.SetArgument(4, y_buffer());
  kernel1.SetArgument(5, static_cast<int>(y_offset));
  kernel1.SetArgument(6, static_cast<int>(y_inc));
  kernel1.SetArgument(7, temp_buffer());
  kernel1.SetArgument(8, static_cast<int>(do_conjugate));

  // Event waiting list
  auto eventWaitList = std::vector<Event>();

  // Launches the main kernel
  auto global1 = std::vector<size_t>{db_["WGS1"]*temp_size};
  auto local1 = std::vector<size_t>{db_["WGS1"]};
  auto kernelEvent = Event();
  RunKernel(kernel1, queue_, device_, global1, local1, kernelEvent.pointer());
  eventWaitList.push_back(kernelEvent);

  // Sets the arguments for the epilogue kernel
  kernel2.SetArgument(0, temp_buffer());
  kernel2.SetArgument(1, dot_buffer());
  kernel2.SetArgument(2, static_cast<int>(dot_offset));

  // Launches the epilogue kernel
  auto global2 = std::vector<size_t>{db_["WGS2"]};
  auto local2 = std::vector<size_t>{db_["WGS2"]};
  RunKernel(kernel2, queue_, device_, global2, local2, event_, eventWaitList);
}

// =================================================================================================

// Compiles the templated class
template class Xdot<half>;
template class Xdot<float>;
template class Xdot<double>;
template class Xdot<float2>;
template class Xdot<double2>;

// =================================================================================================
} // namespace clblast
