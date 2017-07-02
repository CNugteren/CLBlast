
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xim2col class (see the header for information about the class).
//
// =================================================================================================

#include "routines/levelx/xim2col.hpp"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xim2col<T>::Xim2col(Queue &queue, EventPointer event, const std::string &name):
        Routine(queue, event, name, {}, PrecisionValue<T>(), {}, {
#include "../../kernels/level3/level3.opencl"
        }) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xim2col<T>::DoIm2col(const size_t channels, const size_t height, const size_t width,
                          const size_t kernel_h, const size_t kernel_w, const size_t pad_h,
                          const size_t pad_w, const size_t stride_h, const size_t stride_w,
                          const size_t dilation_h, const size_t dilation_w,
                          const Buffer<T> &im_buffer, const size_t im_offset,
                          const Buffer<T> &col_buffer, const size_t col_offset) {

  // Makes sure all dimensions are larger than zero
  if ((channels == 0) || (height == 0) || (width == 0)) { throw BLASError(StatusCode::kInvalidDimension); }
}

// =================================================================================================

// Compiles the templated class
template class Xim2col<half>;
template class Xim2col<float>;
template class Xim2col<double>;
template class Xim2col<float2>;
template class Xim2col<double2>;

// =================================================================================================
} // namespace clblast
