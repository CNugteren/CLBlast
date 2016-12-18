
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xtrsm class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level3/xtrsm.hpp"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xtrsm<T>::Xtrsm(Queue &queue, EventPointer event, const std::string &name):
    Xgemm<T>(queue, event, name) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xtrsm<T>::DoTrsm(const Layout layout, const Side side, const Triangle triangle,
                      const Transpose a_transpose, const Diagonal diagonal,
                      const size_t m, const size_t n,
                      const T alpha,
                      const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                      const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld) {

  // Makes sure all dimensions are larger than zero
  if ((m == 0) || (n == 0)) { throw BLASError(StatusCode::kInvalidDimension); }

  // Computes the k dimension. This is based on whether or not matrix is A (on the left)
  // or B (on the right) in the Xgemm routine.
  auto k = (side == Side::kLeft) ? m : n;

  // Checks for validity of the triangular A matrix
  TestMatrixA(k, k, a_buffer, a_offset, a_ld);

  // Checks for validity of the input/output B matrix
  const auto b_one = (layout == Layout::kRowMajor) ? n : m;
  const auto b_two = (layout == Layout::kRowMajor) ? m : n;
  TestMatrixB(b_one, b_two, b_buffer, b_offset, b_ld);

  // Creates a copy of B to avoid overwriting input in GEMM while computing output
  const auto b_size = (b_ld * (b_two - 1) + b_one + b_offset);
  auto b_buffer_copy = Buffer<T>(context_, b_size);
  b_buffer.CopyTo(queue_, b_size, b_buffer_copy);

  // TODO: Implement TRSM computation
}

// =================================================================================================

// Compiles the templated class
template class Xtrsm<half>;
template class Xtrsm<float>;
template class Xtrsm<double>;
template class Xtrsm<float2>;
template class Xtrsm<double2>;

// =================================================================================================
} // namespace clblast
