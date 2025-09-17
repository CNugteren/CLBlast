
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xsymv routine. It is based on the generalized mat-vec multiplication
// routine (Xgemv). The Xsymv class inherits from the templated class Xgemv, allowing it to call the
// "MatVec" function directly.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XSYMV_H_
#define CLBLAST_ROUTINES_XSYMV_H_

#include <cstddef>
#include <string>

#include "clblast.h"
#include "routines/level2/xgemv.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xsymv : public Xgemv<T> {
 public:
  // Uses the generic matrix-vector routine
  using Xgemv<T>::MatVec;

  // Constructor
  Xsymv(Queue& queue, EventPointer event, const std::string& name = "SYMV");

  // Templated-precision implementation of the routine
  void DoSymv(const Layout layout, const Triangle triangle, const size_t n, const T alpha, const Buffer<T>& a_buffer,
              const size_t a_offset, const size_t a_ld, const Buffer<T>& x_buffer, const size_t x_offset,
              const size_t x_inc, const T beta, const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XSYMV_H_
#endif
