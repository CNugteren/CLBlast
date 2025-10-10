
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
  void DoIm2col(KernelMode kernel_mode, size_t channels, size_t height, size_t width, size_t kernel_h, size_t kernel_w,
                size_t pad_h, size_t pad_w, size_t stride_h, size_t stride_w, size_t dilation_h, size_t dilation_w,
                const Buffer<T>& im_buffer, size_t im_offset, const Buffer<T>& col_buffer, size_t col_offset);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XIM2COL_H_
#endif
