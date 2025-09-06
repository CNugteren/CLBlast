
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xspr class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level2/xspr.hpp"

#include <cstddef>
#include <string>

#include "clblast.h"
#include "routines/level2/xher.hpp"
#include "utilities/backend.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xspr<T>::Xspr(Queue& queue, EventPointer event, const std::string& name) : Xher<T, T>(queue, event, name) {}

// =================================================================================================

// The main routine
template <typename T>
void Xspr<T>::DoSpr(const Layout layout, const Triangle triangle, const size_t n, const T alpha,
                    const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc, const Buffer<T>& ap_buffer,
                    const size_t ap_offset) {
  // Specific Xspr functionality is implemented in the kernel using defines
  DoHer(layout, triangle, n, alpha, x_buffer, x_offset, x_inc, ap_buffer, ap_offset, n,
        true);  // packed matrix
}

// =================================================================================================

// Compiles the templated class
template class Xspr<half>;
template class Xspr<float>;
template class Xspr<double>;

// =================================================================================================
}  // namespace clblast
