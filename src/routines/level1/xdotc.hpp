
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xdotc routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XDOTC_H_
#define CLBLAST_ROUTINES_XDOTC_H_

#include "routines/level1/xdot.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xdotc : public Xdot<T> {
 public:
  // Uses the regular Xdot routine
  using Xdot<T>::DoDot;

  // Constructor
  Xdotc(Queue& queue, EventPointer event, const std::string& name = "DOTC");

  // Templated-precision implementation of the routine
  void DoDotc(const size_t n, const Buffer<T>& dot_buffer, const size_t dot_offset, const Buffer<T>& x_buffer,
              const size_t x_offset, const size_t x_inc, const Buffer<T>& y_buffer, const size_t y_offset,
              const size_t y_inc);
};
extern template class Xdotc<float2>;
extern template class Xdotc<double2>;


// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XDOTC_H_
#endif
