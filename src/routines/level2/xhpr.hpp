
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xhpr routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XHPR_H_
#define CLBLAST_ROUTINES_XHPR_H_

#include "routines/level2/xher.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T, typename U>
class Xhpr: public Xher<T,U> {
 public:

  // Uses the regular Xher routine
  using Xher<T,U>::DoHer;

  // Constructor
  Xhpr(Queue &queue, EventPointer event, const std::string &name = "HPR");

  // Templated-precision implementation of the routine
  void DoHpr(const Layout layout, const Triangle triangle,
             const size_t n,
             const U alpha,
             const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
             const Buffer<T> &ap_buffer, const size_t ap_offset);
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_XHPR_H_
#endif
