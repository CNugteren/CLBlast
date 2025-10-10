
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xtrmm routine. The implementation is based on first transforming the
// upper/lower unit/non-unit triangular matrix into a regular matrix and then calling the GEMM
// routine. Therefore, this class inherits from the Xgemm class.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XTRMM_H_
#define CLBLAST_ROUTINES_XTRMM_H_

#include <cstddef>
#include <string>

#include "routines/level3/xgemm.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xtrmm : public Xgemm<T> {
 public:
  // Uses methods and variables the regular Xgemm routine
  using Xgemm<T>::getRoutineName;
  using Xgemm<T>::getQueue;
  using Xgemm<T>::getContext;
  using Xgemm<T>::getDevice;
  using Xgemm<T>::getProgram;
  using Xgemm<T>::getDatabase;
  using Xgemm<T>::DoGemm;

  // Constructor
  Xtrmm(Queue& queue, EventPointer event, const std::string& name = "TRMM");

  // Templated-precision implementation of the routine
  void DoTrmm(Layout layout, Side side, Triangle triangle, Transpose a_transpose, Diagonal diagonal, size_t m, size_t n,
              T alpha, const Buffer<T>& a_buffer, size_t a_offset, size_t a_ld, const Buffer<T>& b_buffer,
              size_t b_offset, size_t b_ld);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XTRMM_H_
#endif
