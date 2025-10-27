
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xsyr2k routine. The precision is implemented using a template argument.
// The implementation is very similar to Xsyrk (see header for details), except for the fact that
// the main XgemmUpper/XgemmLower kernel is called twice: C = AB^T + C and C = BA^T + C.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XSYR2K_H_
#define CLBLAST_ROUTINES_XSYR2K_H_

#include <cstddef>
#include <string>

#include "routines/level3/xsyrk.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xsyr2k : public Xsyrk<T> {
 public:
  // Uses methods and variables the regular Xsyrk routine
  using Xsyrk<T>::getEvent;
  using Xsyrk<T>::SyrkAB;

  // Constructor
  Xsyr2k(Queue& queue, EventPointer event, const std::string& name = "SYR2K");

  // Templated-precision implementation of the routine
  void DoSyr2k(Layout layout, Triangle triangle, Transpose ab_transpose, size_t n, size_t k, T alpha,
               const Buffer<T>& a_buffer, size_t a_offset, size_t a_ld, const Buffer<T>& b_buffer, size_t b_offset,
               size_t b_ld, T beta, const Buffer<T>& c_buffer, size_t c_offset, size_t c_ld);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XSYR2K_H_
#endif
