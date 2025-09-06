
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xher2 routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XHER2_H_
#define CLBLAST_ROUTINES_XHER2_H_

#include <cstddef>
#include <string>

#include "clblast.h"
#include "routine.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xher2 : public Routine {
 public:
  // Constructor
  Xher2(Queue& queue, EventPointer event, const std::string& name = "HER2");

  // Templated-precision implementation of the routine
  void DoHer2(Layout layout, Triangle triangle, size_t n, T alpha, const Buffer<T>& x_buffer, size_t x_offset,
              size_t x_inc, const Buffer<T>& y_buffer, size_t y_offset, size_t y_inc, const Buffer<T>& a_buffer,
              size_t a_offset, size_t a_ld, bool packed = false);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XHER2_H_
#endif
