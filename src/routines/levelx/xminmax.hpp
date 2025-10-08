
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Ekansh Jain
//
// This file implements the Xminmax routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XMINMAX_H_
#define CLBLAST_ROUTINES_XMINMAX_H_

#include <cstddef>
#include <string>

#include "routine.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xminmax : public Routine {
 public:
  // Constructor
  Xminmax(Queue& queue, EventPointer event, const std::string& name = "MINMAX");

  // Templated-precision implementation of the routine
  void DoMinmax(const size_t n, const Buffer<unsigned int>& imax_buffer, const size_t imax_offset,
                const Buffer<unsigned int>& imin_buffer, const size_t imin_offset, const Buffer<T>& x_buffer,
                const size_t x_offset, const size_t x_inc);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XMINMAX_H_
#endif
