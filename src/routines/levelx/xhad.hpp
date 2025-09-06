
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xhad routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XHAD_H_
#define CLBLAST_ROUTINES_XHAD_H_

#include <cstddef>
#include <string>

#include "routine.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xhad : public Routine {
 public:
  // Constructor
  Xhad(Queue& queue, EventPointer event, const std::string& name = "HAD");

  // Templated-precision implementation of the routine
  void DoHad(size_t n, T alpha, const Buffer<T>& x_buffer, size_t x_offset, size_t x_inc, const Buffer<T>& y_buffer,
             size_t y_offset, size_t y_inc, T beta, const Buffer<T>& z_buffer, size_t z_offset, size_t z_inc);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XHAD_H_
#endif
