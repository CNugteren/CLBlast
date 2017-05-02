
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the XaxpyBatched routine. This is a non-blas batched version of AXPY.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XAXPYBATCHED_H_
#define CLBLAST_ROUTINES_XAXPYBATCHED_H_

#include <vector>

#include "routine.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class XaxpyBatched: public Routine {
 public:

  // Constructor
  XaxpyBatched(Queue &queue, EventPointer event, const std::string &name = "AXPYBATCHED");

  // Templated-precision implementation of the routine
  void DoAxpyBatched(const size_t n, const std::vector<T> &alphas,
                     const Buffer<T> &x_buffer, const std::vector<size_t> &x_offsets, const size_t x_inc,
                     const Buffer<T> &y_buffer, const std::vector<size_t> &y_offsets, const size_t y_inc,
                     const size_t batch_count);
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_XAXPYBATCHED_H_
#endif
