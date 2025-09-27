
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xim2col routine. The precision is implemented using a template argument.
// Uses the tuning parameters from the regular copy kernel.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XIM2COL_H_
#define CLBLAST_ROUTINES_XIM2COL_H_

#include <cstddef>
#include <string>

#include "routine.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xim2col : public Routine {
 public:
  // Constructor
  Xim2col(Queue& queue, EventPointer event, const std::string& name = "IM2COL");

  // Templated-precision implementation of the routine
  void DoIm2col(const KernelMode kernel_mode, const size_t channels, const size_t height, const size_t width,
                const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w,
                const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                const Buffer<T>& im_buffer, const size_t im_offset, const Buffer<T>& col_buffer,
                const size_t col_offset);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XIM2COL_H_
#endif
