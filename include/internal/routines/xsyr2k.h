
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xsyr2k routine. The precision is implemented using a template argument.
// The implementation is very similar to Xsyrk (see header for details), except for the fact that
// the main XgemmUpper/XgemmLower kernel is called twice: C = AB^T + C and C = BA^T + C.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XSYR2K_H_
#define CLBLAST_ROUTINES_XSYR2K_H_

#include "internal/routine.h"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xsyr2k: public Routine {
 public:
  Xsyr2k(CommandQueue &queue, Event &event);

  // Templated-precision implementation of the routine
  StatusCode DoSyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                     const size_t n, const size_t k,
                     const T alpha,
                     const Buffer &a_buffer, const size_t a_offset, const size_t a_ld,
                     const Buffer &b_buffer, const size_t b_offset, const size_t b_ld,
                     const T beta,
                     const Buffer &c_buffer, const size_t c_offset, const size_t c_ld);

 private:
  // Static variable to get the precision
  const static Precision precision_;
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_XSYR2K_H_
#endif
