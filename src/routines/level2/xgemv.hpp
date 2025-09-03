
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xgemv routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XGEMV_H_
#define CLBLAST_ROUTINES_XGEMV_H_

#include "routine.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xgemv : public Routine {
 public:
  // Constructor
  Xgemv(Queue& queue, EventPointer event, const std::string& name = "GEMV");

  // Templated-precision implementation of the routine
  void DoGemv(const Layout layout, const Transpose a_transpose, const size_t m, const size_t n, const T alpha,
              const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld, const Buffer<T>& x_buffer,
              const size_t x_offset, const size_t x_inc, const T beta, const Buffer<T>& y_buffer, const size_t y_offset,
              const size_t y_inc);

  // Generic version used also for other matrix-vector multiplications
  void MatVec(const Layout layout, const Transpose a_transpose, const size_t m, const size_t n, const T alpha,
              const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld, const Buffer<T>& x_buffer,
              const size_t x_offset, const size_t x_inc, const T beta, const Buffer<T>& y_buffer, const size_t y_offset,
              const size_t y_inc, bool fast_kernel, bool fast_kernel_rot, const size_t parameter, const bool packed,
              const size_t kl, const size_t ku);
};

extern template class Xgemv<half>;
extern template class Xgemv<float>;
extern template class Xgemv<double>;
extern template class Xgemv<float2>;
extern template class Xgemv<double2>;
// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XGEMV_H_
#endif
