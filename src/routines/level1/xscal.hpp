
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xscal routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XSCAL_H_
#define CLBLAST_ROUTINES_XSCAL_H_

#include "routine.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xscal: public Routine {
 public:

  // Constructor
  Xscal(Queue &queue, EventPointer event, const std::string &name = "SCAL");

  // Templated-precision implementation of the routine
  void DoScal(const size_t n, const T alpha,
              const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc);
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_XSCAL_H_
#endif
