
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xspr2 routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XSPR2_H_
#define CLBLAST_ROUTINES_XSPR2_H_

#include "routines/level2/xher2.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xspr2 : public Xher2<T> {
 public:
  // Uses the regular Xher2 routine
  using Xher2<T>::DoHer2;

  // Constructor
  Xspr2(Queue& queue, EventPointer event, const std::string& name = "SPR2");

  // Templated-precision implementation of the routine
  void DoSpr2(const Layout layout, const Triangle triangle, const size_t n, const T alpha, const Buffer<T>& x_buffer,
              const size_t x_offset, const size_t x_inc, const Buffer<T>& y_buffer, const size_t y_offset,
              const size_t y_inc, const Buffer<T>& ap_buffer, const size_t ap_offset);
};

extern template class Xspr2<half>;
extern template class Xspr2<float>;
extern template class Xspr2<double>;
extern template class Xspr2<float2>;
extern template class Xspr2<double2>;
// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XSPR2_H_
#endif
