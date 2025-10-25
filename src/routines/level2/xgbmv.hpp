
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xgbmv routine. It is based on the generalized mat-vec multiplication
// routine (Xgemv). The Xgbmv class inherits from the templated class Xgemv, allowing it to call the
// "MatVec" function directly.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XGBMV_H_
#define CLBLAST_ROUTINES_XGBMV_H_

#include <cstddef>
#include <string>

#include "routines/level2/xgemv.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xgbmv : public Xgemv<T> {
 public:
  // Uses the generic matrix-vector routine
  using Xgemv<T>::MatVec;

  // Constructor
  Xgbmv(Queue& queue, EventPointer event, const std::string& name = "GBMV");

  // Templated-precision implementation of the routine
  void DoGbmv(Layout layout, Transpose a_transpose, size_t m, size_t n, size_t kl, size_t ku, T alpha,
              const Buffer<T>& a_buffer, size_t a_offset, size_t a_ld, const Buffer<T>& x_buffer, size_t x_offset,
              size_t x_inc, T beta, const Buffer<T>& y_buffer, size_t y_offset, size_t y_inc);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XGBMV_H_
#endif
