
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xamax routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XAMAX_H_
#define CLBLAST_ROUTINES_XAMAX_H_

#include "internal/routine.h"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xamax: public Routine<T> {
 public:

  // Members and methods from the base class
  using Routine<T>::db_;
  using Routine<T>::source_string_;
  using Routine<T>::queue_;
  using Routine<T>::event_;
  using Routine<T>::context_;
  using Routine<T>::GetProgramFromCache;
  using Routine<T>::RunKernel;
  using Routine<T>::ErrorIn;

  // Constructor
  Xamax(Queue &queue, EventPointer event, const std::string &name = "AMAX");

  // Templated-precision implementation of the routine
  StatusCode DoAmax(const size_t n,
                    const Buffer<unsigned int> &imax_buffer, const size_t imax_offset,
                    const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc);

 private:
  // Static variable to get the precision
  const static Precision precision_;
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_XAMAX_H_
#endif
