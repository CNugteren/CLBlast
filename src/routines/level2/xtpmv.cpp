
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xtpmv class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level2/xtpmv.hpp"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xtpmv<T>::Xtpmv(Queue &queue, EventPointer event, const std::string &name):
    Xgemv<T>(queue, event, name) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xtpmv<T>::DoTpmv(const Layout layout, const Triangle triangle,
                      const Transpose a_transpose, const Diagonal diagonal,
                      const size_t n,
                      const Buffer<T> &ap_buffer, const size_t ap_offset,
                      const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc) {

  // Creates a copy of X: a temporary scratch buffer
  auto scratch_buffer = Buffer<T>(context_, n*x_inc + x_offset);
  x_buffer.CopyTo(queue_, n*x_inc + x_offset, scratch_buffer);

  // The data is either in the upper or lower triangle
  size_t is_upper = ((triangle == Triangle::kUpper && layout != Layout::kRowMajor) ||
                     (triangle == Triangle::kLower && layout == Layout::kRowMajor));

  // Adds '2' to the parameter if the diagonal is unit
  auto parameter = (diagonal == Diagonal::kUnit) ? is_upper + 2 : is_upper;

  // Runs the generic matrix-vector multiplication, disabling the use of fast vectorized kernels.
  // The specific triangular packed matrix-accesses are implemented in the kernel guarded by the
  // ROUTINE_TPMV define.
  auto fast_kernels = false;
  try {
    MatVec(layout, a_transpose,
           n, n, ConstantOne<T>(),
           ap_buffer, ap_offset, n,
           scratch_buffer, x_offset, x_inc, ConstantZero<T>(),
           x_buffer, x_offset, x_inc,
           fast_kernels, fast_kernels,
           parameter, true, 0, 0);
  } catch (BLASError &e) {
    // Returns the proper error code (renames vector Y to X)
    switch (e.status()) {
      case StatusCode::kInvalidVectorY:      throw BLASError(StatusCode::kInvalidVectorX, e.details());
      case StatusCode::kInvalidIncrementY:   throw BLASError(StatusCode::kInvalidIncrementX, e.details());
      case StatusCode::kInsufficientMemoryY: throw BLASError(StatusCode::kInsufficientMemoryX, e.details());
      default:                               throw;
    }
  }
}

// =================================================================================================

// Compiles the templated class
template class Xtpmv<half>;
template class Xtpmv<float>;
template class Xtpmv<double>;
template class Xtpmv<float2>;
template class Xtpmv<double2>;

// =================================================================================================
} // namespace clblast
