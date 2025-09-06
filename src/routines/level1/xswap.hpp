
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xswap routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XSWAP_H_
#define CLBLAST_ROUTINES_XSWAP_H_

#include <cstddef>
#include <string>

#include "routine.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xswap : public Routine {
 public:
  // Constructor
  Xswap(Queue& queue, EventPointer event, const std::string& name = "SWAP");

  // Templated-precision implementation of the routine
  void DoSwap(size_t n, const Buffer<T>& x_buffer, size_t x_offset, size_t x_inc, const Buffer<T>& y_buffer,
              size_t y_offset, size_t y_inc);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XSWAP_H_
#endif
