
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

#include "routines/level2/xgemv.hpp"

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
  void DoGbmv(const Layout layout, const Transpose a_transpose, const size_t m, const size_t n, const size_t kl,
              const size_t ku, const T alpha, const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
              const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc, const T beta,
              const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc);
};
extern template class Xgbmv<half>;
extern template class Xgbmv<float>;
extern template class Xgbmv<double>;
extern template class Xgbmv<float2>;
extern template class Xgbmv<double2>;


// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XGBMV_H_
#endif
