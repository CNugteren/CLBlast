
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xdotc class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level1/xdotc.hpp"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xdotc<T>::Xdotc(Queue &queue, EventPointer event, const std::string &name):
    Xdot<T>(queue, event, name) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xdotc<T>::DoDotc(const size_t n,
                      const Buffer<T> &dot_buffer, const size_t dot_offset,
                      const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                      const Buffer<T> &y_buffer, const size_t y_offset, const size_t y_inc) {
  DoDot(n, dot_buffer, dot_offset,
        x_buffer, x_offset, x_inc,
        y_buffer, y_offset, y_inc,
        true);
}

// =================================================================================================

// Compiles the templated class
template class Xdotc<float2>;
template class Xdotc<double2>;

// =================================================================================================
} // namespace clblast
