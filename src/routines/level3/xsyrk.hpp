
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xsyrk routine. The precision is implemented using a template argument.
// The implementation is based on the regular Xgemm routine and kernel, but with two main changes:
// 1) The final unpad(transpose) kernel updates only the upper/lower triangular part.
// 2) The main Xgemm kernel masks workgroups not contributing to usefull data. This is only for
//    performance reasons, as the actual masking is done later (see the first point).
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XSYRK_H_
#define CLBLAST_ROUTINES_XSYRK_H_

#include "routine.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xsyrk: public Routine {
 public:

  // Constructor
  Xsyrk(Queue &queue, EventPointer event, const std::string &name = "SYRK");

  // Templated-precision implementation of the routine
  void DoSyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
              const size_t n, const size_t k,
              const T alpha,
              const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
              const T beta,
              const Buffer<T> &c_buffer, const size_t c_offset, const size_t c_ld);

  // Helper function to be reused for SYR2K
  void SyrkAB(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Transpose b_transpose,
              const size_t n, const size_t k,
              const T alpha,
              const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
              const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld,
              const T beta,
              const Buffer<T> &c_buffer, const size_t c_offset, const size_t c_ld,
              EventPointer final_event);
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_XSYRK_H_
#endif
