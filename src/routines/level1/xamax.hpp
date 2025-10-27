
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xamax routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XAMAX_H_
#define CLBLAST_ROUTINES_XAMAX_H_

#include <cstddef>
#include <string>

#include "routine.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xamax : public Routine {
 public:
  // Constructor
  Xamax(Queue& queue, EventPointer event, const std::string& name = "AMAX");

  // Templated-precision implementation of the routine
  void DoAmax(size_t n, const Buffer<unsigned int>& imax_buffer, size_t imax_offset, const Buffer<T>& x_buffer,
              size_t x_offset, size_t x_inc);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XAMAX_H_
#endif
