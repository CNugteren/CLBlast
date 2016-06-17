
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xnrm2 routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XNRM2_H_
#define CLBLAST_ROUTINES_XNRM2_H_

#include "internal/routine.h"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xnrm2: public Routine<T> {
 public:

  // Members and methods from the base class
  using Routine<T>::db_;
  using Routine<T>::source_string_;
  using Routine<T>::queue_;
  using Routine<T>::device_;
  using Routine<T>::event_;
  using Routine<T>::context_;
  using Routine<T>::routine_name_;

  // Constructor
  Xnrm2(Queue &queue, EventPointer event, const std::string &name = "NRM2");

  // Templated-precision implementation of the routine
  StatusCode DoNrm2(const size_t n,
                    const Buffer<T> &nrm2_buffer, const size_t nrm2_offset,
                    const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc);
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_XNRM2_H_
#endif
