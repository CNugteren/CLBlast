
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xcopy routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XCOPY_H_
#define CLBLAST_ROUTINES_XCOPY_H_

#include "routine.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xcopy : public Routine {
 public:
  // Constructor
  Xcopy(Queue& queue, EventPointer event, const std::string& name = "COPY");

  // Templated-precision implementation of the routine
  void DoCopy(size_t n, const Buffer<T>& x_buffer, size_t x_offset, size_t x_inc, const Buffer<T>& y_buffer,
              size_t y_offset, size_t y_inc);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XCOPY_H_
#endif
