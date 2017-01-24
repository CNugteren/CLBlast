
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xasum class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level1/xasum.hpp"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xasum<T>::Xasum(Queue &queue, EventPointer event, const std::string &name):
    Routine(queue, event, name, {"Xdot"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level1/xasum.opencl"
    }) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xasum<T>::DoAsum(const size_t n,
                      const Buffer<T> &asum_buffer, const size_t asum_offset,
                      const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc) {

  // Makes sure all dimensions are larger than zero
  if (n == 0) { throw BLASError(StatusCode::kInvalidDimension); }

  // Tests the vectors for validity
  TestVectorX(n, x_buffer, x_offset, x_inc);
  TestVectorScalar(1, asum_buffer, asum_offset);

  // Retrieves the Xasum kernels from the compiled binary
  auto kernel1 = Kernel(program_, "Xasum");
  auto kernel2 = Kernel(program_, "XasumEpilogue");

  // Creates the buffer for intermediate values
  auto temp_size = 2*db_["WGS2"];
  auto temp_buffer = Buffer<T>(context_, temp_size);

  // Sets the kernel arguments
  kernel1.SetArgument(0, static_cast<int>(n));
  kernel1.SetArgument(1, x_buffer());
  kernel1.SetArgument(2, static_cast<int>(x_offset));
  kernel1.SetArgument(3, static_cast<int>(x_inc));
  kernel1.SetArgument(4, temp_buffer());

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
  kernel2.SetArgument(1, asum_buffer());
  kernel2.SetArgument(2, static_cast<int>(asum_offset));

  // Launches the epilogue kernel
  auto global2 = std::vector<size_t>{db_["WGS2"]};
  auto local2 = std::vector<size_t>{db_["WGS2"]};
  RunKernel(kernel2, queue_, device_, global2, local2, event_, eventWaitList);
}

// =================================================================================================

// Compiles the templated class
template class Xasum<half>;
template class Xasum<float>;
template class Xasum<double>;
template class Xasum<float2>;
template class Xasum<double2>;

// =================================================================================================
} // namespace clblast
