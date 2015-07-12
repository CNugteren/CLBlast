
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xhemm routine. It is based on the generalized matrix multiplication
// routine (Xgemm). The implementation is very similar to the Xsymm routine.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XHEMM_H_
#define CLBLAST_ROUTINES_XHEMM_H_

#include "internal/routines/level3/xgemm.h"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xhemm: public Xgemm<T> {
 public:

  // Uses several variables from the Routine class
  using Routine::db_;
  using Routine::context_;

  // Uses several helper functions from the Routine class
  using Routine::RunKernel;
  using Routine::ErrorIn;
  using Routine::TestMatrixA;
  using Routine::GetProgramFromCache;

  // Uses the regular Xgemm routine
  using Xgemm<T>::DoGemm;

  // Constructor
  Xhemm(CommandQueue &queue, Event &event);

  // Templated-precision implementation of the routine
  StatusCode DoHemm(const Layout layout, const Side side, const Triangle triangle,
                    const size_t m, const size_t n,
                    const T alpha,
                    const Buffer &a_buffer, const size_t a_offset, const size_t a_ld,
                    const Buffer &b_buffer, const size_t b_offset, const size_t b_ld,
                    const T beta,
                    const Buffer &c_buffer, const size_t c_offset, const size_t c_ld);
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_XHEMM_H_
#endif
