
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xherk routine. The precision is implemented using the template argument
// 'T', whereas the alpha/beta arguments are of type 'U'. The implementation is very similar to the
// Xsyrk routine.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XHERK_H_
#define CLBLAST_ROUTINES_XHERK_H_

#include <cstddef>
#include <string>

#include "routine.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T, typename U>
class Xherk : public Routine {
 public:
  // Constructor
  Xherk(Queue& queue, EventPointer event, const std::string& name = "HERK");

  // Templated-precision implementation of the routine
  void DoHerk(Layout layout, Triangle triangle, Transpose a_transpose, size_t n, size_t k, U alpha,
              const Buffer<T>& a_buffer, size_t a_offset, size_t a_ld, U beta, const Buffer<T>& c_buffer,
              size_t c_offset, size_t c_ld);

  // Helper function to be reused for HER2K
  void HerkAB(Layout layout, Triangle triangle, Transpose a_transpose, Transpose b_transpose, size_t n, size_t k,
              T complex_alpha, const Buffer<T>& a_buffer, size_t a_offset, size_t a_ld, const Buffer<T>& b_buffer,
              size_t b_offset, size_t b_ld, T complex_beta, const Buffer<T>& c_buffer, size_t c_offset, size_t c_ld,
              EventPointer final_event, bool diagonal_to_zero);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XHERK_H_
#endif
