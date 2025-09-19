
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xtrsv routine. It uses a block-algorithm and performs small triangular
// forward and backward substitutions on the diagonal parts of the matrix in combination with larger
// GEMV computation on the remainder of the matrix.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XTRSV_H_
#define CLBLAST_ROUTINES_XTRSV_H_

#include <cstddef>
#include <string>

#include "clblast.h"
#include "routines/level2/xgemv.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xtrsv : public Xgemv<T> {
 public:
  // Uses the generic matrix-vector routine
  using Xgemv<T>::getQueue;
  using Xgemv<T>::getContext;
  using Xgemv<T>::getDevice;
  using Xgemv<T>::getDatabase;
  using Xgemv<T>::getProgram;
  using Xgemv<T>::getEvent;
  using Xgemv<T>::DoGemv;

  // Constructor
  Xtrsv(Queue& queue, EventPointer event, const std::string& name = "TRSV");

  // Templated-precision implementation of the routine
  void DoTrsv(Layout layout, Triangle triangle, Transpose a_transpose, Diagonal diagonal, size_t n,
              const Buffer<T>& a_buffer, size_t a_offset, size_t a_ld, const Buffer<T>& b_buffer, size_t b_offset,
              size_t b_inc);

  // Performs forward or backward substitution on a small triangular matrix
  void Substitution(Layout layout, Triangle triangle, Transpose a_transpose, Diagonal diagonal, size_t n,
                    const Buffer<T>& a_buffer, size_t a_offset, size_t a_ld, const Buffer<T>& b_buffer, size_t b_offset,
                    size_t b_inc, const Buffer<T>& x_buffer, size_t x_offset, size_t x_inc, EventPointer event);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XTRSV_H_
#endif
