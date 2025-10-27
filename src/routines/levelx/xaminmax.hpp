
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Ekansh Jain
//
// This file implements the Xaminmax routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XAMINMAX_H_
#define CLBLAST_ROUTINES_XAMINMAX_H_

#include <cstddef>
#include <string>

#include "routines/levelx/xminmax.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xaminmax : public Xminmax<T> {
 public:
  // Members and methods from the base class
  using Xminmax<T>::DoMinmax;

  // Constructor
  Xaminmax(Queue& queue, EventPointer event, const std::string& name = "AMINMAX") : Xminmax<T>(queue, event, name) {}

  // Forwards to the regular non-absolute version. The implementation difference is realised in the
  // kernel through a pre-processor macro based on the name of the routine.
  void DoAminmax(const size_t n, const Buffer<unsigned int>& imax_buffer, const size_t imax_offset,
                 const Buffer<unsigned int>& imin_buffer, const size_t imin_offset, const Buffer<T>& x_buffer,
                 const size_t x_offset, const size_t x_inc) {
    DoMinmax(n, imax_buffer, imax_offset, imin_buffer, imin_offset, x_buffer, x_offset, x_inc);
  }
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XAMINMAX_H_
#endif