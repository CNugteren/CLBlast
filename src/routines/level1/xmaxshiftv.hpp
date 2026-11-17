
// =================================================================================================
// This file is part of the CLBlast project.
//
// This file implements the Xmaxshiftv routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XMAXSHIFTV_H_
#define CLBLAST_ROUTINES_XMAXSHIFTV_H_

#include <cstddef>
#include <string>

#include "routine.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

template <typename T>
class Xmaxshiftv : public Routine {
 public:
  // Constructor
  Xmaxshiftv(Queue& queue, EventPointer event, const std::string& name = "MAXSHIFTV");

  // Subtracts the maximum from every element of the vector in-place
  void DoMaxshiftv(const size_t n, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XMAXSHIFTV_H_
#endif
