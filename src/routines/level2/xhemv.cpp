
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xhemv class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level2/xhemv.hpp"

#include <cstddef>
#include <string>

#include "clblast.h"
#include "routines/level2/xgemv.hpp"
#include "utilities/backend.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xhemv<T>::Xhemv(Queue& queue, EventPointer event, const std::string& name) : Xgemv<T>(queue, event, name) {}

// =================================================================================================

// The main routine
template <typename T>
void Xhemv<T>::DoHemv(const Layout layout, const Triangle triangle, const size_t n, const T alpha,
                      const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld, const Buffer<T>& x_buffer,
                      const size_t x_offset, const size_t x_inc, const T beta, const Buffer<T>& y_buffer,
                      const size_t y_offset, const size_t y_inc) {
  // The data is either in the upper or lower triangle
  size_t is_upper = ((triangle == Triangle::kUpper && layout != Layout::kRowMajor) ||
                     (triangle == Triangle::kLower && layout == Layout::kRowMajor));

  // Runs the generic matrix-vector multiplication, disabling the use of fast vectorized kernels.
  // The specific hermitian matrix-accesses are implemented in the kernel guarded by the
  // ROUTINE_HEMV define.
  bool fast_kernels = false;
  MatVec(layout, Transpose::kNo, n, n, alpha, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, beta, y_buffer,
         y_offset, y_inc, fast_kernels, fast_kernels, is_upper, false, 0, 0);
}

// =================================================================================================

// Compiles the templated class
template class Xhemv<float2>;
template class Xhemv<double2>;

// =================================================================================================
}  // namespace clblast
