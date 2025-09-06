
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xspr2 routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XSPR2_H_
#define CLBLAST_ROUTINES_XSPR2_H_

#include <cstddef>
#include <string>

#include "clblast.h"
#include "routines/level2/xher2.hpp"
#include "utilities/backend.hpp"

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
  void DoSpr2(Layout layout, Triangle triangle, size_t n, T alpha, const Buffer<T>& x_buffer, size_t x_offset,
              size_t x_inc, const Buffer<T>& y_buffer, size_t y_offset, size_t y_inc, const Buffer<T>& ap_buffer,
              size_t ap_offset);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XSPR2_H_
#endif
