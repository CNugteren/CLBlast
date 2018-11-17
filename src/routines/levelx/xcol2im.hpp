
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xcol2im routine. The precision is implemented using a template argument.
// Uses the tuning parameters from the regular copy kernel.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XCOL2IM_H_
#define CLBLAST_ROUTINES_XCOL2IM_H_

#include "routine.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xcol2im: public Routine {
 public:

  // Constructor
  Xcol2im(Queue &queue, EventPointer event, const std::string &name = "COL2IM");

  // Templated-precision implementation of the routine
  void DoCol2im(const KernelMode kernel_mode,
                const size_t channels, const size_t height, const size_t width,
                const size_t kernel_h, const size_t kernel_w,
                const size_t pad_h, const size_t pad_w,
                const size_t stride_h, const size_t stride_w,
                const size_t dilation_h, const size_t dilation_w,
                const Buffer<T> &col_buffer, const size_t col_offset,
                const Buffer<T> &im_buffer, const size_t im_offset);
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_XCOL2IM_H_
#endif
