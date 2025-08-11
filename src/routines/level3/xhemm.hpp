
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

#include "routines/level3/xgemm.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xhemm : public Xgemm<T> {
 public:
  // Uses methods and variables the regular Xgemm routine
  using Xgemm<T>::routine_name_;
  using Xgemm<T>::queue_;
  using Xgemm<T>::context_;
  using Xgemm<T>::device_;
  using Xgemm<T>::program_;
  using Xgemm<T>::db_;
  using Xgemm<T>::DoGemm;

  // Constructor
  Xhemm(Queue& queue, EventPointer event, const std::string& name = "HEMM");

  // Templated-precision implementation of the routine
  void DoHemm(const Layout layout, const Side side, const Triangle triangle, const size_t m, const size_t n,
              const T alpha, const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
              const Buffer<T>& b_buffer, const size_t b_offset, const size_t b_ld, const T beta,
              const Buffer<T>& c_buffer, const size_t c_offset, const size_t c_ld);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XHEMM_H_
#endif
