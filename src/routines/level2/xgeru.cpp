
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xgeru class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level2/xgeru.hpp"

#include <cstddef>
#include <string>

#include "routines/level2/xger.hpp"
#include "utilities/backend.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xgeru<T>::Xgeru(Queue& queue, EventPointer event, const std::string& name) : Xger<T>(queue, event, name) {}

// =================================================================================================

// The main routine
template <typename T>
void Xgeru<T>::DoGeru(const Layout layout, const size_t m, const size_t n, const T alpha, const Buffer<T>& x_buffer,
                      const size_t x_offset, const size_t x_inc, const Buffer<T>& y_buffer, const size_t y_offset,
                      const size_t y_inc, const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld) {
  // Regular Ger operation on complex data
  DoGer(layout, m, n, alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, a_buffer, a_offset, a_ld);
}

// =================================================================================================

// Compiles the templated class
template class Xgeru<float2>;
template class Xgeru<double2>;

// =================================================================================================
}  // namespace clblast
