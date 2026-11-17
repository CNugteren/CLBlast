
// =================================================================================================
// This file is part of the CLBlast project.
//
// This file implements the Xmaxshiftm class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level3/xmaxshiftm.hpp"

#include <cstddef>
#include <string>

#include "routines/common.hpp"
#include "utilities/buffer_test.hpp"
#include "utilities/clblast_exceptions.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Constructor: uses the same reduction tuning parameters as Xamax and Xdot
template <typename T>
Xmaxshiftm<T>::Xmaxshiftm(Queue& queue, EventPointer event, const std::string& name)
    : Routine(queue, event, name, {"Xdot"}, PrecisionValue<T>(), {},
              {
#include "../../kernels/level3/xmaxshiftm.opencl"
              }) {
}

// =================================================================================================

template <typename T>
void Xmaxshiftm<T>::DoMaxshiftm(const Layout layout, const size_t m, const size_t n, const Buffer<T>& a_buffer,
                                const size_t a_offset, const size_t a_ld) {
  if (m == 0 || n == 0) {
    throw BLASError(StatusCode::kInvalidDimension);
  }

  const auto is_rowmajor = (layout == Layout::kRowMajor);
  TestMatrixA(is_rowmajor ? n : m, is_rowmajor ? m : n, a_buffer, a_offset, a_ld);

  auto kernel = Kernel(program_, "Xmaxshiftm");
  kernel.SetArgument(0, static_cast<int>(m));
  kernel.SetArgument(1, static_cast<int>(n));
  kernel.SetArgument(2, a_buffer());
  kernel.SetArgument(3, static_cast<int>(a_offset));
  kernel.SetArgument(4, static_cast<int>(a_ld));
  kernel.SetArgument(5, static_cast<int>(is_rowmajor));

  // One workgroup reduces and shifts each row
  RunKernel(kernel, queue_, device_, {db_["WGS1"], m}, {db_["WGS1"], 1}, event_);
}

// =================================================================================================

template class Xmaxshiftm<half>;
template class Xmaxshiftm<float>;
template class Xmaxshiftm<double>;

// =================================================================================================
}  // namespace clblast
