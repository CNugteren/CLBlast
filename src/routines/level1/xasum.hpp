
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xasum routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XASUM_H_
#define CLBLAST_ROUTINES_XASUM_H_

#include "routine.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xasum : public Routine {
 public:
  // Constructor
  Xasum(Queue& queue, EventPointer event, const std::string& name = "ASUM");

  // Templated-precision implementation of the routine
  void DoAsum(size_t n, const Buffer<T>& asum_buffer, size_t asum_offset, const Buffer<T>& x_buffer, size_t x_offset,
              size_t x_inc);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XASUM_H_
#endif
