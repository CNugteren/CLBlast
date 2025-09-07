
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Ekansh Jain
//
// This file implements the Xminmax class (see the header for information about the class).
//
// =================================================================================================

#include "routines/levelx/xminmax.hpp"

#include <string>
#include <vector>

namespace clblast {
template <typename T>
Xminmax<T>::Xminmax(Queue& queue, EventPointer event, const std::string& name)
    : Routine(queue, event, name, {"Xdot"}, PrecisionValue<T>(), {},
              {
#include "../../kernels/levelx/xminmax.opencl"
              }) {
}

template <typename T>
void Xminmax<T>::DoMinmax(const size_t n, const Buffer<unsigned int>& imax_buffer, const size_t imax_offset,
                          const Buffer<unsigned int>& imin_buffer, const size_t imin_offset, const Buffer<T>& x_buffer,
                          const size_t x_offset, const size_t x_inc) {
  // Makes sure all dimensions are larger than zero
  if (n == 0) {
    throw BLASError(StatusCode::kInvalidDimension);
  }

  // Tests the vectors for validity
  TestVectorX(n, x_buffer, x_offset, x_inc);
  TestVectorIndex(1, imax_buffer, imax_offset);
  TestVectorIndex(1, imin_buffer, imin_offset);

  // Retrieves the Xminmax kernels from the compiled binary
  auto kernel1 = Kernel(program_, "Xminmax");
  auto kernel2 = Kernel(program_, "XminmaxEpilogue");

  // Creates the buffer for intermediate values
  auto temp_size = 4 * db_["WGS2"];
  auto temp_buffer1 = Buffer<T>(context_, temp_size);
  auto temp_buffer2 = Buffer<unsigned int>(context_, temp_size);

  // Sets the kernel arguments
  kernel1.SetArgument(0, static_cast<int>(n));
  kernel1.SetArgument(1, x_buffer());
  kernel1.SetArgument(2, static_cast<int>(x_offset));
  kernel1.SetArgument(3, static_cast<int>(x_inc));
  kernel1.SetArgument(4, temp_buffer1());
  kernel1.SetArgument(5, temp_buffer2());

  // Event waiting list
  auto eventWaitList = std::vector<Event>();

  // Launches the main kernel
  auto global1 = std::vector<size_t>{(db_["WGS1"] * temp_size) / 2};
  auto local1 = std::vector<size_t>{db_["WGS1"]};
  auto kernelEvent = Event();
  RunKernel(kernel1, queue_, device_, global1, local1, kernelEvent.pointer());
  eventWaitList.push_back(kernelEvent);

  // Sets the arguments for the epilogue kernel
  kernel2.SetArgument(0, temp_buffer1());
  kernel2.SetArgument(1, temp_buffer2());
  kernel2.SetArgument(2, imax_buffer());
  kernel2.SetArgument(3, static_cast<int>(imax_offset));
  kernel2.SetArgument(4, imin_buffer());
  kernel2.SetArgument(5, static_cast<int>(imin_offset));

  // Launches the epilogue kernel
  auto global2 = std::vector<size_t>{db_["WGS2"]};
  auto local2 = std::vector<size_t>{db_["WGS2"]};
  RunKernel(kernel2, queue_, device_, global2, local2, event_, eventWaitList);
}

template class Xminmax<half>;
template class Xminmax<float>;
template class Xminmax<double>;
template class Xminmax<float2>;
template class Xminmax<double2>;
}  // namespace clblast