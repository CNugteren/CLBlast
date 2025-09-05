
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xasum class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level1/xasum.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "clblast.h"
#include "routine.hpp"
#include "routines/common.hpp"
#include "utilities/backend.hpp"
#include "utilities/buffer_test.hpp"
#include "utilities/clblast_exceptions.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xasum<T>::Xasum(Queue& queue, EventPointer event, const std::string& name)
    : Routine(queue, event, name, {"Xdot"}, PrecisionValue<T>(), {},
              {
#include "../../kernels/level1/xasum.opencl"
              }) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xasum<T>::DoAsum(const size_t n, const Buffer<T>& asum_buffer, const size_t asum_offset, const Buffer<T>& x_buffer,
                      const size_t x_offset, const size_t x_inc) {
  // Makes sure all dimensions are larger than zero
  if (n == 0) {
    throw BLASError(StatusCode::kInvalidDimension);
  }

  // Tests the vectors for validity
  TestVectorX(n, x_buffer, x_offset, x_inc);
  TestVectorScalar(1, asum_buffer, asum_offset);

  // Retrieves the Xasum kernels from the compiled binary
  auto kernel1 = Kernel(getProgram(), "Xasum");
  auto kernel2 = Kernel(getProgram(), "XasumEpilogue");

  // Creates the buffer for intermediate values
  auto temp_size = 2 * getDatabase()["WGS2"];
  auto temp_buffer = Buffer<T>(getContext(), temp_size);

  // Sets the kernel arguments
  kernel1.SetArgument(0, static_cast<int>(n));
  kernel1.SetArgument(1, x_buffer());
  kernel1.SetArgument(2, static_cast<int>(x_offset));
  kernel1.SetArgument(3, static_cast<int>(x_inc));
  kernel1.SetArgument(4, temp_buffer());

  // Event waiting list
  auto eventWaitList = std::vector<Event>();

  // Launches the main kernel
  auto global1 = std::vector<size_t>{getDatabase()["WGS1"] * temp_size};
  auto local1 = std::vector<size_t>{getDatabase()["WGS1"]};
  auto kernelEvent = Event();
  RunKernel(kernel1, getQueue(), getDevice(), global1, local1, kernelEvent.pointer());
  eventWaitList.push_back(kernelEvent);

  // Sets the arguments for the epilogue kernel
  kernel2.SetArgument(0, temp_buffer());
  kernel2.SetArgument(1, asum_buffer());
  kernel2.SetArgument(2, static_cast<int>(asum_offset));

  // Launches the epilogue kernel
  auto global2 = std::vector<size_t>{getDatabase()["WGS2"]};
  auto local2 = std::vector<size_t>{getDatabase()["WGS2"]};
  RunKernel(kernel2, getQueue(), getDevice(), global2, local2, getEvent(), eventWaitList);
}

// =================================================================================================

// Compiles the templated class
template class Xasum<half>;
template class Xasum<float>;
template class Xasum<double>;
template class Xasum<float2>;
template class Xasum<double2>;

// =================================================================================================
}  // namespace clblast
