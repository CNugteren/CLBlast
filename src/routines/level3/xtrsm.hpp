
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xtrsm routine. The implementation is based on ??? (TODO).
// Therefore, this class inherits from the Xgemm class.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XTRSM_H_
#define CLBLAST_ROUTINES_XTRSM_H_

#include "routines/level3/xgemm.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xtrsm: public Xgemm<T> {
 public:

  // Uses methods and variables the Xgemm routine
  using Xgemm<T>::queue_;
  using Xgemm<T>::context_;
  using Xgemm<T>::device_;
  using Xgemm<T>::db_;
  using Xgemm<T>::program_;
  using Xgemm<T>::event_;
  using Xgemm<T>::DoGemm;

  // Constructor
  Xtrsm(Queue &queue, EventPointer event, const std::string &name = "TRSM");

  // Templated-precision implementation of the routine
  void DoTrsm(const Layout layout, Side side, Triangle triangle,
              const Transpose a_transpose, const Diagonal diagonal,
              size_t m, size_t n,
              const T alpha,
              const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
              const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld);

  // Implementation of the column-major version
  void TrsmColMajor(const Side side, const Triangle triangle,
                    const Transpose a_transpose, const Diagonal diagonal,
                    const size_t m, const size_t n,
                    const T alpha,
                    const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                    const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld);
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_XTRSM_H_
#endif
