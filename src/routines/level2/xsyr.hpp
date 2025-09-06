
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xsyr routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XSYR_H_
#define CLBLAST_ROUTINES_XSYR_H_

#include <cstddef>
#include <string>

#include "clblast.h"
#include "routines/level2/xher.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xsyr : public Xher<T, T> {
 public:
  // Uses the regular Xher routine
  using Xher<T, T>::DoHer;

  // Constructor
  Xsyr(Queue& queue, EventPointer event, const std::string& name = "SYR");

  // Templated-precision implementation of the routine
  void DoSyr(Layout layout, Triangle triangle, size_t n, T alpha, const Buffer<T>& x_buffer,
             size_t x_offset, size_t x_inc, const Buffer<T>& a_buffer, size_t a_offset,
             size_t a_ld);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XSYR_H_
#endif
