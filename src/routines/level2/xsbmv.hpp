
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xsbmv routine. It is based on the generalized mat-vec multiplication
// routine (Xgemv). The Xsbmv class inherits from the templated class Xgemv, allowing it to call the
// "MatVec" function directly.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XSBMV_H_
#define CLBLAST_ROUTINES_XSBMV_H_

#include "routines/level2/xgemv.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xsbmv : public Xgemv<T> {
 public:
  // Uses the generic matrix-vector routine
  using Xgemv<T>::MatVec;

  // Constructor
  Xsbmv(Queue& queue, EventPointer event, const std::string& name = "SBMV");

  // Templated-precision implementation of the routine
  void DoSbmv(Layout layout, Triangle triangle, size_t n, size_t k, T alpha, const Buffer<T>& a_buffer, size_t a_offset,
              size_t a_ld, const Buffer<T>& x_buffer, size_t x_offset, size_t x_inc, T beta, const Buffer<T>& y_buffer,
              size_t y_offset, size_t y_inc);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XSBMV_H_
#endif
