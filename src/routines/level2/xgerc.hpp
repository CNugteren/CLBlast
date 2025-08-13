
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xgerc routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XGERC_H_
#define CLBLAST_ROUTINES_XGERC_H_

#include "routines/level2/xger.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xgerc : public Xger<T> {
 public:
  // Uses the regular Xger routine
  using Xger<T>::DoGer;

  // Constructor
  Xgerc(Queue& queue, EventPointer event, const std::string& name = "GERC");

  // Templated-precision implementation of the routine
  void DoGerc(const Layout layout, const size_t m, const size_t n, const T alpha, const Buffer<T>& x_buffer,
              const size_t x_offset, const size_t x_inc, const Buffer<T>& y_buffer, const size_t y_offset,
              const size_t y_inc, const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XGERC_H_
#endif
