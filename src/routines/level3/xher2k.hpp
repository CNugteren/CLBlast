
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xher2k routine. The precision is implemented using the template argument
// 'T', whereas the alpha/beta arguments are of type 'U'. The implementation is very similar to the
// Xsyr2k routine.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XHER2K_H_
#define CLBLAST_ROUTINES_XHER2K_H_

#include <cstddef>
#include <string>

#include "clblast.h"
#include "routines/level3/xherk.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T, typename U>
class Xher2k : public Xherk<T, U> {
 public:
  // Uses methods and variables the regular Xherk routine
  using Xherk<T, U>::getEvent;
  using Xherk<T, U>::HerkAB;

  // Constructor
  Xher2k(Queue& queue, EventPointer event, const std::string& name = "HER2K");

  // Templated-precision implementation of the routine
  void DoHer2k(Layout layout, Triangle triangle, Transpose ab_transpose, size_t n, size_t k, T alpha,
               const Buffer<T>& a_buffer, size_t a_offset, size_t a_ld, const Buffer<T>& b_buffer, size_t b_offset,
               size_t b_ld, U beta, const Buffer<T>& c_buffer, size_t c_offset, size_t c_ld);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XHER2K_H_
#endif
