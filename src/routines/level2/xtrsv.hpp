
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xtrsv routine. It uses a block-algorithm and performs small triangular
// forward and backward substitutions on the diagonal parts of the matrix in combination with larger
// GEMV computation on the remainder of the matrix.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XTRSV_H_
#define CLBLAST_ROUTINES_XTRSV_H_

#include "routines/level2/xgemv.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xtrsv: public Xgemv<T> {
 public:

  // Uses the generic matrix-vector routine
  using Xgemv<T>::queue_;
  using Xgemv<T>::context_;
  using Xgemv<T>::device_;
  using Xgemv<T>::db_;
  using Xgemv<T>::program_;
  using Xgemv<T>::event_;
  using Xgemv<T>::DoGemv;

  // Constructor
  Xtrsv(Queue &queue, EventPointer event, const std::string &name = "TRSV");

  // Templated-precision implementation of the routine
  void DoTrsv(const Layout layout, const Triangle triangle,
              const Transpose a_transpose, const Diagonal diagonal,
              const size_t n,
              const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
              const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc);

  // Performs forward or backward substitution on a small triangular matrix
  void Substitution(const Layout layout, const Triangle triangle,
                    const Transpose a_transpose, const Diagonal diagonal,
                    const size_t n,
                    const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                    const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_inc,
                    const Buffer<T> &x_buffer, const size_t offset_x, const size_t x_inc,
                    EventPointer event);
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_XTRSV_H_
#endif
