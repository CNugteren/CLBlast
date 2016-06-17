
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xher routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XHER_H_
#define CLBLAST_ROUTINES_XHER_H_

#include "internal/routine.h"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T, typename U>
class Xher: public Routine<T> {
 public:

  // Members and methods from the base class
  using Routine<T>::db_;
  using Routine<T>::source_string_;
  using Routine<T>::queue_;
  using Routine<T>::event_;
  using Routine<T>::context_;
  using Routine<T>::GetProgramFromCache;
  using Routine<T>::RunKernel;

  // Constructor
  Xher(Queue &queue, EventPointer event, const std::string &name = "HER");

  // Translates alpha of type 'U' into type 'T'
  T GetAlpha(const U alpha);

  // Templated-precision implementation of the routine
  StatusCode DoHer(const Layout layout, const Triangle triangle,
                   const size_t n,
                   const U alpha,
                   const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                   const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                   const bool packed = false);

 private:
  // Static variable to get the precision
  const static Precision precision_;
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_XHER_H_
#endif
