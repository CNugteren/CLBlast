
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
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
class Xtrsm : public Xgemm<T> {
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
  Xtrsm(Queue& queue, EventPointer event, const std::string& name = "TRSM");

  // Templated-precision implementation of the routine
  void DoTrsm(Layout layout, Side side, Triangle triangle, Transpose a_transpose, Diagonal diagonal, size_t m, size_t n,
              T alpha, const Buffer<T>& a_buffer, size_t a_offset, size_t a_ld, const Buffer<T>& b_buffer,
              size_t b_offset, size_t b_ld);

  // Implementation of the column-major version
  void TrsmColMajor(Side side, Triangle triangle, Transpose a_transpose, Diagonal diagonal, size_t m, size_t n, T alpha,
                    const Buffer<T>& a_buffer, size_t a_offset, size_t a_ld, const Buffer<T>& b_buffer, size_t b_offset,
                    size_t b_ld);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XTRSM_H_
#endif
