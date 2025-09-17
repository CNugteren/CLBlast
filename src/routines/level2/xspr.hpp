
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xspr routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XSPR_H_
#define CLBLAST_ROUTINES_XSPR_H_

#include <cstddef>
#include <string>


#include "routines/level2/xher.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xspr : public Xher<T, T> {
 public:
  // Uses the regular Xher routine
  using Xher<T, T>::DoHer;

  // Constructor
  Xspr(Queue& queue, EventPointer event, const std::string& name = "SPR");

  // Templated-precision implementation of the routine
  void DoSpr(const Layout layout, const Triangle triangle, const size_t n, const T alpha, const Buffer<T>& x_buffer,
             const size_t x_offset, const size_t x_inc, const Buffer<T>& ap_buffer, const size_t ap_offset);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XSPR_H_
#endif
