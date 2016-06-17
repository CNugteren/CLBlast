
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xher2k routine. The precision is implemented using the template argument
// 'T', whereas the alpha/beta arguments are of type 'U'. The implementation is very similar to the
// Xsyr2k routine.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XHER2K_H_
#define CLBLAST_ROUTINES_XHER2K_H_

#include "internal/routine.h"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T, typename U>
class Xher2k: public Routine<T> {
 public:

  // Members and methods from the base class
  using Routine<T>::db_;
  using Routine<T>::source_string_;
  using Routine<T>::queue_;
  using Routine<T>::event_;
  using Routine<T>::context_;
  using Routine<T>::GetProgramFromCache;
  using Routine<T>::PadCopyTransposeMatrix;
  using Routine<T>::RunKernel;

  // Constructor
  Xher2k(Queue &queue, EventPointer event, const std::string &name = "HER2K");

  // Templated-precision implementation of the routine
  StatusCode DoHer2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                     const size_t n, const size_t k,
                     const T alpha,
                     const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                     const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld,
                     const U beta,
                     const Buffer<T> &c_buffer, const size_t c_offset, const size_t c_ld);

 private:
  // Static variable to get the precision
  const static Precision precision_;
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_XHER2K_H_
#endif
