
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xhemm routine. It is based on the generalized matrix multiplication
// routine (Xgemm). The implementation is very similar to the Xsymm routine.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XHEMM_H_
#define CLBLAST_ROUTINES_XHEMM_H_

#include <cstddef>
#include <string>

#include "clblast.h"
#include "routines/level3/xgemm.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xhemm : public Xgemm<T> {
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
  Xhemm(Queue& queue, EventPointer event, const std::string& name = "HEMM");

  // Templated-precision implementation of the routine
  void DoHemm(Layout layout, Side side, Triangle triangle, size_t m, size_t n, T alpha, const Buffer<T>& a_buffer,
              size_t a_offset, size_t a_ld, const Buffer<T>& b_buffer, size_t b_offset, size_t b_ld, T beta,
              const Buffer<T>& c_buffer, size_t c_offset, size_t c_ld);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XHEMM_H_
#endif
