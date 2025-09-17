
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xher routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XHER_H_
#define CLBLAST_ROUTINES_XHER_H_

#include <cstddef>
#include <string>


#include "routine.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T, typename U>
class Xher : public Routine {
 public:
  // Constructor
  Xher(Queue& queue, EventPointer event, const std::string& name = "HER");

  // Translates alpha of type 'U' into type 'T'
  T GetAlpha(const U alpha);

  // Templated-precision implementation of the routine
  void DoHer(const Layout layout, const Triangle triangle, const size_t n, const U alpha, const Buffer<T>& x_buffer,
             const size_t x_offset, const size_t x_inc, const Buffer<T>& a_buffer, const size_t a_offset,
             const size_t a_ld, const bool packed = false);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XHER_H_
#endif
