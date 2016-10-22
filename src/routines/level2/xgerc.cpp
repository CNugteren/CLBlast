
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xgerc class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level2/xgerc.hpp"

#include <string>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xgerc<T>::Xgerc(Queue &queue, EventPointer event, const std::string &name):
    Xger<T>(queue, event, name) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xgerc<T>::DoGerc(const Layout layout,
                      const size_t m, const size_t n,
                      const T alpha,
                      const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                      const Buffer<T> &y_buffer, const size_t y_offset, const size_t y_inc,
                      const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld) {

  // Regular Ger operation on complex data, plus conjugation in the kernel guarded by the
  // ROUTINE_GERC guard.
  DoGer(layout, m, n, alpha,
        x_buffer, x_offset, x_inc,
        y_buffer, y_offset, y_inc,
        a_buffer, a_offset, a_ld);
}

// =================================================================================================

// Compiles the templated class
template class Xgerc<float2>;
template class Xgerc<double2>;

// =================================================================================================
} // namespace clblast
