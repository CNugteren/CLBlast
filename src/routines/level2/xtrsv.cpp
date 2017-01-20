
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xtrsv class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level2/xtrsv.hpp"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xtrsv<T>::Xtrsv(Queue &queue, EventPointer event, const std::string &name):
    Xgemv<T>(queue, event, name) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xtrsv<T>::DoTrsv(const Layout layout, const Triangle triangle,
                      const Transpose a_transpose, const Diagonal diagonal,
                      const size_t n,
                      const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                      const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc) {

  // Makes sure all dimensions are larger than zero
  if (n == 0) { throw BLASError(StatusCode::kInvalidDimension); }

  // Tests the matrix and vector
  TestMatrixA(n, n, a_buffer, a_offset, a_ld);
  TestVectorX(n, x_buffer, x_offset, x_inc);

  // Creates a copy of X: a temporary scratch buffer
  auto scratch_buffer = Buffer<T>(context_, n*x_inc + x_offset);
  x_buffer.CopyTo(queue_, n*x_inc + x_offset, scratch_buffer);

  // The data is either in the upper or lower triangle
  size_t is_upper = ((triangle == Triangle::kUpper && layout != Layout::kRowMajor) ||
                     (triangle == Triangle::kLower && layout == Layout::kRowMajor));

  // TODO: Implement the routine
}

// =================================================================================================

// Compiles the templated class
template class Xtrsv<half>;
template class Xtrsv<float>;
template class Xtrsv<double>;
template class Xtrsv<float2>;
template class Xtrsv<double2>;

// =================================================================================================
} // namespace clblast
