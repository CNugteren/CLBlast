
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xomatcopy routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XOMATCOPY_H_
#define CLBLAST_ROUTINES_XOMATCOPY_H_

#include "routine.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xomatcopy: public Routine {
 public:

  // Constructor
  Xomatcopy(Queue &queue, EventPointer event, const std::string &name = "OMATCOPY");

  // Helper functions to query what kind of kernel will run
  size_t GetVectorWidth(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n,
                        const size_t a_offset, const size_t a_ld,
                        const size_t b_offset, const size_t b_ld) {
    const auto transpose = (a_transpose != Transpose::kNo);
    const auto conjugate = (a_transpose == Transpose::kConjugate);
    const auto rotated = (layout == Layout::kRowMajor);
    const auto a_one = (rotated) ? n : m;
    const auto a_two = (rotated) ? m : n;
    const auto b_one = (transpose) ? a_two : a_one;
    const auto b_two = (transpose) ? a_one : a_two;
    size_t vector_width = 1;
    if (transpose) {
      if (UseFastTransposeKernel(db_, a_one, a_two, a_ld, a_offset, b_one, b_two,
                                 b_ld, b_offset, conjugate, false, false, false)) {
        vector_width = db_["TRA_WPT"];
      }
    }
    else {
      if (UseFastCopyKernel(db_, a_one, a_two, a_ld, a_offset, b_one, b_two,
                            b_ld, b_offset, conjugate, false, false, false)) {
        vector_width = db_["COPY_VW"];
      }
    }
    if (vector_width > 4) { throw RuntimeError("Image support not available for vectorwidth > 4"); }
    return vector_width;
  }

  // Templated-precision implementation of the routine
  void DoOmatcopy(const Layout layout, const Transpose a_transpose,
                  const size_t m, const size_t n, const T alpha,
                  const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                  const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld);
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_XOMATCOPY_H_
#endif
