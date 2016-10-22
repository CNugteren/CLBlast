
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xmin routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XMIN_H_
#define CLBLAST_ROUTINES_XMIN_H_

#include "routine.hpp"
#include "routines/level1/xamax.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xmin: public Xamax<T> {
 public:

  // Members and methods from the base class
  using Xamax<T>::DoAmax;

  // Constructor
  Xmin(Queue &queue, EventPointer event, const std::string &name = "MIN"):
    Xamax<T>(queue, event, name) {
  }

  // Forwards to the regular max-absolute version. The implementation difference is realised in the
  // kernel through a pre-processor macro based on the name of the routine.
  void DoMin(const size_t n,
             const Buffer<unsigned int> &imin_buffer, const size_t imin_offset,
             const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc) {
    DoAmax(n, imin_buffer, imin_offset, x_buffer, x_offset, x_inc);
  }
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_XMIN_H_
#endif
