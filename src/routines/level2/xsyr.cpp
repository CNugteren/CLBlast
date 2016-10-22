
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xsyr class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level2/xsyr.hpp"

#include <string>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xsyr<T>::Xsyr(Queue &queue, EventPointer event, const std::string &name):
    Xher<T,T>(queue, event, name) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xsyr<T>::DoSyr(const Layout layout, const Triangle triangle,
                    const size_t n,
                    const T alpha,
                    const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                    const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld) {

  // Specific Xsyr functionality is implemented in the kernel using defines
  DoHer(layout, triangle, n, alpha,
        x_buffer, x_offset, x_inc,
        a_buffer, a_offset, a_ld);
}

// =================================================================================================

// Compiles the templated class
template class Xsyr<half>;
template class Xsyr<float>;
template class Xsyr<double>;

// =================================================================================================
} // namespace clblast
