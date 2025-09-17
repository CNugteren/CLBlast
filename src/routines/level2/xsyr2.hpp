
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xsyr2 routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XSYR2_H_
#define CLBLAST_ROUTINES_XSYR2_H_

#include <cstddef>
#include <string>

#include "clblast.h"
#include "routines/level2/xher2.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xsyr2 : public Xher2<T> {
 public:
  // Uses the regular Xher2 routine
  using Xher2<T>::DoHer2;

  // Constructor
  Xsyr2(Queue& queue, EventPointer event, const std::string& name = "SYR2");

  // Templated-precision implementation of the routine
  void DoSyr2(const Layout layout, const Triangle triangle, const size_t n, const T alpha, const Buffer<T>& x_buffer,
              const size_t x_offset, const size_t x_inc, const Buffer<T>& y_buffer, const size_t y_offset,
              const size_t y_inc, const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XSYR2_H_
#endif
