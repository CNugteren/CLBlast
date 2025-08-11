
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xsyr2 class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level2/xsyr2.hpp"

#include <string>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xsyr2<T>::Xsyr2(Queue &queue, EventPointer event, const std::string &name):
    Xher2<T>(queue, event, name) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xsyr2<T>::DoSyr2(const Layout layout, const Triangle triangle,
                      const size_t n,
                      const T alpha,
                      const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                      const Buffer<T> &y_buffer, const size_t y_offset, const size_t y_inc,
                      const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld) {

  // Specific Xsyr2 functionality is implemented in the kernel using defines
  DoHer2(layout, triangle, n, alpha,
         x_buffer, x_offset, x_inc,
         y_buffer, y_offset, y_inc,
         a_buffer, a_offset, a_ld);
}

// =================================================================================================

// Compiles the templated class
template class Xsyr2<half>;
template class Xsyr2<float>;
template class Xsyr2<double>;

// =================================================================================================
} // namespace clblast
