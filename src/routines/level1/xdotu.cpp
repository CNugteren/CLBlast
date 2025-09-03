
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xdotu class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level1/xdotu.hpp"

#include <cstddef>
#include <string>

#include "routines/level1/xdot.hpp"
#include "utilities/backend.hpp"
#include "utilities/utilities.hpp"


namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xdotu<T>::Xdotu(Queue& queue, EventPointer event, const std::string& name) : Xdot<T>(queue, event, name) {}

// =================================================================================================

// The main routine
template <typename T>
void Xdotu<T>::DoDotu(const size_t n, const Buffer<T>& dot_buffer, const size_t dot_offset, const Buffer<T>& x_buffer,
                      const size_t x_offset, const size_t x_inc, const Buffer<T>& y_buffer, const size_t y_offset,
                      const size_t y_inc) {
  DoDot(n, dot_buffer, dot_offset, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, false);
}

// =================================================================================================

// Compiles the templated class
template class Xdotu<float2>;
template class Xdotu<double2>;

// =================================================================================================
}  // namespace clblast
