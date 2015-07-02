
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xtrmm routine. The implementation is based on first transforming the
// upper/lower unit/non-unit triangular matrix into a regular matrix and then calling the GEMM
// routine. Therefore, this class inherits from the Xgemm class.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XTRMM_H_
#define CLBLAST_ROUTINES_XTRMM_H_

#include "internal/routines/xgemm.h"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xtrmm: public Xgemm<T> {
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
  Xtrmm(CommandQueue &queue, Event &event);

  // Templated-precision implementation of the routine
  StatusCode DoTrmm(const Layout layout, const Side side, const Triangle triangle,
                    const Transpose a_transpose, const Diagonal diagonal,
                    const size_t m, const size_t n,
                    const T alpha,
                    const Buffer &a_buffer, const size_t a_offset, const size_t a_ld,
                    const Buffer &b_buffer, const size_t b_offset, const size_t b_ld);
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_XTRMM_H_
#endif
