
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xtbmv routine. It is based on the generalized mat-vec multiplication
// routine (Xgemv). The Xtbmv class inherits from the templated class Xgemv, allowing it to call the
// "MatVec" function directly.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XTBMV_H_
#define CLBLAST_ROUTINES_XTBMV_H_

#include <cstddef>
#include <string>

#include "routines/level2/xgemv.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xtbmv : public Xgemv<T> {
 public:
  // Uses the generic matrix-vector routine
  using Xgemv<T>::getQueue;
  using Xgemv<T>::getContext;
  using Xgemv<T>::MatVec;

  // Constructor
  Xtbmv(Queue& queue, EventPointer event, const std::string& name = "TBMV");

  // Templated-precision implementation of the routine
  void DoTbmv(Layout layout, Triangle triangle, Transpose a_transpose, Diagonal diagonal, size_t n, size_t k,
              const Buffer<T>& a_buffer, size_t a_offset, size_t a_ld, const Buffer<T>& x_buffer, size_t x_offset,
              size_t x_inc);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XTBMV_H_
#endif
