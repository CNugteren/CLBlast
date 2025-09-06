
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xgemv routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XGEMV_H_
#define CLBLAST_ROUTINES_XGEMV_H_

#include <cstddef>
#include <string>

#include "clblast.h"
#include "routine.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xgemv : public Routine {
 public:
  // Constructor
  Xgemv(Queue& queue, EventPointer event, const std::string& name = "GEMV");

  // Templated-precision implementation of the routine
  void DoGemv(Layout layout, Transpose a_transpose, size_t m, size_t n, T alpha, const Buffer<T>& a_buffer,
              size_t a_offset, size_t a_ld, const Buffer<T>& x_buffer, size_t x_offset, size_t x_inc, T beta,
              const Buffer<T>& y_buffer, size_t y_offset, size_t y_inc);

  // Generic version used also for other matrix-vector multiplications
  void MatVec(Layout layout, Transpose a_transpose, size_t m, size_t n, T alpha, const Buffer<T>& a_buffer,
              size_t a_offset, size_t a_ld, const Buffer<T>& x_buffer, size_t x_offset, size_t x_inc, T beta,
              const Buffer<T>& y_buffer, size_t y_offset, size_t y_inc, bool fast_kernel, bool fast_kernel_rot,
              size_t parameter, bool packed, size_t kl, size_t ku);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XGEMV_H_
#endif
