
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xger routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XGER_H_
#define CLBLAST_ROUTINES_XGER_H_

#include "routine.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xger: public Routine {
 public:

  // Constructor
  Xger(Queue &queue, EventPointer event, const std::string &name = "GER");

  // Templated-precision implementation of the routine
  void DoGer(const Layout layout,
             const size_t m, const size_t n,
             const T alpha,
             const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
             const Buffer<T> &y_buffer, const size_t y_offset, const size_t y_inc,
             const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld);
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_XGER_H_
#endif
