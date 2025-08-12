
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xomatcopy routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XOMATCOPY_H_
#define CLBLAST_ROUTINES_XOMATCOPY_H_

#include "routine.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xomatcopy: public Routine {
 public:

  // Constructor
  Xomatcopy(Queue &queue, EventPointer event, const std::string &name = "OMATCOPY");

  // Templated-precision implementation of the routine
  void DoOmatcopy(const Layout layout, const Transpose a_transpose,
                  const size_t m, const size_t n, const T alpha,
                  const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                  const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld);
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_XOMATCOPY_H_
#endif
