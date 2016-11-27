
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xhpr2 class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level2/xhpr2.hpp"

#include <string>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xhpr2<T>::Xhpr2(Queue &queue, EventPointer event, const std::string &name):
    Xher2<T>(queue, event, name) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xhpr2<T>::DoHpr2(const Layout layout, const Triangle triangle,
                      const size_t n,
                      const T alpha,
                      const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                      const Buffer<T> &y_buffer, const size_t y_offset, const size_t y_inc,
                      const Buffer<T> &ap_buffer, const size_t ap_offset) {

  // Specific Xhpr2 functionality is implemented in the kernel using defines
  DoHer2(layout, triangle, n, alpha,
         x_buffer, x_offset, x_inc,
         y_buffer, y_offset, y_inc,
         ap_buffer, ap_offset, n,
         true); // packed matrix
}

// =================================================================================================

// Compiles the templated class
template class Xhpr2<float2>;
template class Xhpr2<double2>;

// =================================================================================================
} // namespace clblast
