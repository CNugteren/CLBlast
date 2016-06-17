
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

#include "internal/routines/level3/xgemm.h"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xtrmm: public Xgemm<T> {
 public:

  // Members and methods from the base class
  using Routine<T>::db_;
  using Routine<T>::queue_;
  using Routine<T>::device_;
  using Routine<T>::context_;
  using Routine<T>::GetProgramFromCache;

  // Uses the regular Xgemm routine
  using Xgemm<T>::DoGemm;

  // Constructor
  Xtrmm(Queue &queue, EventPointer event, const std::string &name = "TRMM");

  // Templated-precision implementation of the routine
  StatusCode DoTrmm(const Layout layout, const Side side, const Triangle triangle,
                    const Transpose a_transpose, const Diagonal diagonal,
                    const size_t m, const size_t n,
                    const T alpha,
                    const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                    const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld);
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_XTRMM_H_
#endif
