
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xsymv class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level2/xsymv.hpp"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xsymv<T>::Xsymv(Queue &queue, EventPointer event, const std::string &name):
    Xgemv<T>(queue, event, name) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xsymv<T>::DoSymv(const Layout layout, const Triangle triangle,
                      const size_t n,
                      const T alpha,
                      const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                      const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                      const T beta,
                      const Buffer<T> &y_buffer, const size_t y_offset, const size_t y_inc) {

  // The data is either in the upper or lower triangle
  size_t is_upper = ((triangle == Triangle::kUpper && layout != Layout::kRowMajor) ||
                     (triangle == Triangle::kLower && layout == Layout::kRowMajor));

  // Runs the generic matrix-vector multiplication, disabling the use of fast vectorized kernels.
  // The specific symmetric matrix-accesses are implemented in the kernel guarded by the
  // ROUTINE_SYMV define.
  bool fast_kernels = false;
  MatVec(layout, Transpose::kNo,
         n, n, alpha,
         a_buffer, a_offset, a_ld,
         x_buffer, x_offset, x_inc, beta,
         y_buffer, y_offset, y_inc,
         fast_kernels, fast_kernels,
         is_upper, false, 0, 0);
}

// =================================================================================================

// Compiles the templated class
template class Xsymv<half>;
template class Xsymv<float>;
template class Xsymv<double>;

// =================================================================================================
} // namespace clblast
