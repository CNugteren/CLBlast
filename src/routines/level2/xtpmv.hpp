
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xtpmv routine. It is based on the generalized mat-vec multiplication
// routine (Xgemv). The Xtpmv class inherits from the templated class Xgemv, allowing it to call the
// "MatVec" function directly.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XTPMV_H_
#define CLBLAST_ROUTINES_XTPMV_H_

#include "routines/level2/xgemv.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xtpmv : public Xgemv<T> {
 public:
  // Uses the generic matrix-vector routine
  using Xgemv<T>::queue_;
  using Xgemv<T>::context_;
  using Xgemv<T>::MatVec;

  // Constructor
  Xtpmv(Queue& queue, EventPointer event, const std::string& name = "TPMV");

  // Templated-precision implementation of the routine
  void DoTpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
              const size_t n, const Buffer<T>& ap_buffer, const size_t ap_offset, const Buffer<T>& x_buffer,
              const size_t x_offset, const size_t x_inc);
};
extern template class Xtpmv<half>;
extern template class Xtpmv<float>;
extern template class Xtpmv<double>;
extern template class Xtpmv<float2>;
extern template class Xtpmv<double2>;


// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XTPMV_H_
#endif
