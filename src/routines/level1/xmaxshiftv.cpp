
// =================================================================================================
// This file is part of the CLBlast project.
//
// This file implements the Xmaxshiftv class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level1/xmaxshiftv.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "routines/common.hpp"
#include "utilities/buffer_test.hpp"
#include "utilities/clblast_exceptions.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Constructor: uses the same reduction tuning parameters as Xamax and Xdot
template <typename T>
Xmaxshiftv<T>::Xmaxshiftv(Queue& queue, EventPointer event, const std::string& name)
    : Routine(queue, event, name, {"Xdot"}, PrecisionValue<T>(), {},
              {
#include "../../kernels/level1/xmaxshiftv.opencl"
              }) {
}

// =================================================================================================

template <typename T>
void Xmaxshiftv<T>::DoMaxshiftv(const size_t n, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc) {
  if (n == 0) {
    throw BLASError(StatusCode::kInvalidDimension);
  }
  TestVectorX(n, x_buffer, x_offset, x_inc);

  auto kernel1 = Kernel(program_, "Xmaxshiftv");
  auto kernel2 = Kernel(program_, "XmaxshiftvEpilogue");
  auto kernel3 = Kernel(program_, "XmaxshiftvShift");

  // Uses at most 2*WGS2 workgroups, as in the other two-stage reductions
  const auto num_groups = std::min(CeilDiv(n, db_["WGS1"]), 2 * db_["WGS2"]);
  auto partial_buffer = Buffer<T>(context_, num_groups);
  auto max_buffer = Buffer<T>(context_, 1);

  kernel1.SetArgument(0, static_cast<int>(n));
  kernel1.SetArgument(1, x_buffer());
  kernel1.SetArgument(2, static_cast<int>(x_offset));
  kernel1.SetArgument(3, static_cast<int>(x_inc));
  kernel1.SetArgument(4, partial_buffer());

  // Explicit dependencies also support out-of-order queues
  auto reduction_event = Event();
  RunKernel(kernel1, queue_, device_, {db_["WGS1"] * num_groups}, {db_["WGS1"]}, reduction_event.pointer());

  kernel2.SetArgument(0, static_cast<int>(num_groups));
  kernel2.SetArgument(1, partial_buffer());
  kernel2.SetArgument(2, max_buffer());
  auto epilogue_event = Event();
  RunKernel(kernel2, queue_, device_, {db_["WGS2"]}, {db_["WGS2"]}, epilogue_event.pointer(), {reduction_event});

  kernel3.SetArgument(0, static_cast<int>(n));
  kernel3.SetArgument(1, x_buffer());
  kernel3.SetArgument(2, static_cast<int>(x_offset));
  kernel3.SetArgument(3, static_cast<int>(x_inc));
  kernel3.SetArgument(4, max_buffer());
  RunKernel(kernel3, queue_, device_, {Ceil(n, db_["WGS2"])}, {db_["WGS2"]}, event_, {epilogue_event});
}

// =================================================================================================

template class Xmaxshiftv<half>;
template class Xmaxshiftv<float>;
template class Xmaxshiftv<double>;

// =================================================================================================
}  // namespace clblast
