
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xhpr class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level2/xhpr.hpp"

#include <string>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T, typename U>
Xhpr<T,U>::Xhpr(Queue &queue, EventPointer event, const std::string &name):
    Xher<T,U>(queue, event, name) {
}

// =================================================================================================

// The main routine
template <typename T, typename U>
void Xhpr<T,U>::DoHpr(const Layout layout, const Triangle triangle,
                      const size_t n,
                      const U alpha,
                      const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                      const Buffer<T> &ap_buffer, const size_t ap_offset) {

  // Specific Xhpr functionality is implemented in the kernel using defines
  DoHer(layout, triangle, n, alpha,
        x_buffer, x_offset, x_inc,
        ap_buffer, ap_offset, n,
        true); // packed matrix
}

// =================================================================================================

// Compiles the templated class
template class Xhpr<float2, float>;
template class Xhpr<double2, double>;

// =================================================================================================
} // namespace clblast
