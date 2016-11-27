
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xgbmv class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level2/xgbmv.hpp"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xgbmv<T>::Xgbmv(Queue &queue, EventPointer event, const std::string &name):
    Xgemv<T>(queue, event, name) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xgbmv<T>::DoGbmv(const Layout layout, const Transpose a_transpose,
                      const size_t m, const size_t n, const size_t kl, const size_t ku,
                      const T alpha,
                      const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                      const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                      const T beta,
                      const Buffer<T> &y_buffer, const size_t y_offset, const size_t y_inc) {

  // Reverses the upper and lower band count
  auto rotated = (layout == Layout::kRowMajor);
  auto kl_real = (rotated) ? ku : kl;
  auto ku_real = (rotated) ? kl : ku;

  // Runs the generic matrix-vector multiplication, disabling the use of fast vectorized kernels.
  // The specific hermitian matrix-accesses are implemented in the kernel guarded by the
  // ROUTINE_GBMV define.
  bool fast_kernels = false;
  MatVec(layout, a_transpose,
         m, n, alpha,
         a_buffer, a_offset, a_ld,
         x_buffer, x_offset, x_inc, beta,
         y_buffer, y_offset, y_inc,
         fast_kernels, fast_kernels,
         0, false, kl_real, ku_real);
}

// =================================================================================================

// Compiles the templated class
template class Xgbmv<half>;
template class Xgbmv<float>;
template class Xgbmv<double>;
template class Xgbmv<float2>;
template class Xgbmv<double2>;

// =================================================================================================
} // namespace clblast
