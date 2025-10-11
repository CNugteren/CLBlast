
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xomatcopy routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XOMATCOPY_H_
#define CLBLAST_ROUTINES_XOMATCOPY_H_

#include <cstddef>
#include <string>

#include "routine.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xomatcopy : public Routine {
 public:
  // Constructor
  Xomatcopy(Queue& queue, EventPointer event, const std::string& name = "OMATCOPY");

  // Templated-precision implementation of the routine
  void DoOmatcopy(Layout layout, Transpose a_transpose, size_t m, size_t n, T alpha, const Buffer<T>& a_buffer,
                  size_t a_offset, size_t a_ld, const Buffer<T>& b_buffer, size_t b_offset, size_t b_ld);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XOMATCOPY_H_
#endif
