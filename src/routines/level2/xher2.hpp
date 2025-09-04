
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xher2 routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XHER2_H_
#define CLBLAST_ROUTINES_XHER2_H_

#include "routine.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xher2 : public Routine {
 public:
  // Constructor
  Xher2(Queue& queue, EventPointer event, const std::string& name = "HER2");

  // Templated-precision implementation of the routine
  void DoHer2(const Layout layout, const Triangle triangle, const size_t n, const T alpha, const Buffer<T>& x_buffer,
              const size_t x_offset, const size_t x_inc, const Buffer<T>& y_buffer, const size_t y_offset,
              const size_t y_inc, const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
              const bool packed = false);
};

extern template class Xher2<half>;
extern template class Xher2<float>;
extern template class Xher2<double>;
extern template class Xher2<float2>;
extern template class Xher2<double2>;

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XHER2_H_
#endif
