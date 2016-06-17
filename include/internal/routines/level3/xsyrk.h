
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

#include "internal/routine.h"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xsyrk: public Routine<T> {
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
  Xsyrk(Queue &queue, EventPointer event, const std::string &name = "SYRK");

  // Templated-precision implementation of the routine
  StatusCode DoSyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                    const size_t n, const size_t k,
                    const T alpha,
                    const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                    const T beta,
                    const Buffer<T> &c_buffer, const size_t c_offset, const size_t c_ld);

 private:
  // Static variable to get the precision
  const static Precision precision_;
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_XSYRK_H_
#endif
