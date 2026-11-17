
// =================================================================================================
// This file is part of the CLBlast project.
//
// This file implements the Xmaxshiftm routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XMAXSHIFTM_H_
#define CLBLAST_ROUTINES_XMAXSHIFTM_H_

#include <cstddef>
#include <string>

#include "routine.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

template <typename T>
class Xmaxshiftm : public Routine {
 public:
  // Constructor
  Xmaxshiftm(Queue& queue, EventPointer event, const std::string& name = "MAXSHIFTM");

  // Subtracts each row's maximum from every element of that row in-place
  void DoMaxshiftm(const Layout layout, const size_t m, const size_t n, const Buffer<T>& a_buffer,
                   const size_t a_offset, const size_t a_ld);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XMAXSHIFTM_H_
#endif
