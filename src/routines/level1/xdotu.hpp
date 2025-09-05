
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xdotu routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XDOTU_H_
#define CLBLAST_ROUTINES_XDOTU_H_

#include <cstddef>
#include <string>

#include "routines/level1/xdot.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xdotu : public Xdot<T> {
 public:
  // Uses the regular Xdot routine
  using Xdot<T>::DoDot;

  // Constructor
  Xdotu(Queue& queue, EventPointer event, const std::string& name = "DOTU");

  // Templated-precision implementation of the routine
  void DoDotu(size_t n, const Buffer<T>& dot_buffer, size_t dot_offset, const Buffer<T>& x_buffer, size_t x_offset,
              size_t x_inc, const Buffer<T>& y_buffer, size_t y_offset, size_t y_inc);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XDOTU_H_
#endif
