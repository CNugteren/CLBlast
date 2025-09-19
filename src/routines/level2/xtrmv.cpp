
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xtrmv class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level2/xtrmv.hpp"

#include <cstddef>
#include <string>

#include "clblast.h"
#include "routines/level2/xgemv.hpp"
#include "utilities/backend.hpp"
#include "utilities/clblast_exceptions.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xtrmv<T>::Xtrmv(Queue& queue, EventPointer event, const std::string& name) : Xgemv<T>(queue, event, name) {}

// =================================================================================================

// The main routine
template <typename T>
void Xtrmv<T>::DoTrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                      const Diagonal diagonal, const size_t n, const Buffer<T>& a_buffer, const size_t a_offset,
                      const size_t a_ld, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc) {
  // Creates a copy of X: a temporary scratch buffer
  const auto x_size = (1 + (n - 1) * x_inc) + x_offset;
  auto scratch_buffer = Buffer<T>(getContext(), x_size);
  x_buffer.CopyTo(getQueue(), x_size, scratch_buffer);

  // The data is either in the upper or lower triangle
  auto is_upper = static_cast<size_t>((triangle == Triangle::kUpper && layout != Layout::kRowMajor) ||
                                      (triangle == Triangle::kLower && layout == Layout::kRowMajor));

  // Adds '2' to the parameter if the diagonal is unit
  auto parameter = (diagonal == Diagonal::kUnit) ? is_upper + 2 : is_upper;

  // Runs the generic matrix-vector multiplication, disabling the use of fast vectorized kernels.
  // The specific triangular matrix-accesses are implemented in the kernel guarded by the
  // ROUTINE_TRMV define.
  auto fast_kernels = false;
  try {
    MatVec(layout, a_transpose, n, n, ConstantOne<T>(), a_buffer, a_offset, a_ld, scratch_buffer, x_offset, x_inc,
           ConstantZero<T>(), x_buffer, x_offset, x_inc, fast_kernels, fast_kernels, parameter, false, 0, 0);
  } catch (BLASError& e) {
    // Returns the proper error code (renames vector Y to X)
    switch (e.status()) {
      case StatusCode::kInvalidVectorY:
        throw BLASError(StatusCode::kInvalidVectorX, e.details());
      case StatusCode::kInvalidIncrementY:
        throw BLASError(StatusCode::kInvalidIncrementX, e.details());
      case StatusCode::kInsufficientMemoryY:
        throw BLASError(StatusCode::kInsufficientMemoryX, e.details());
      default:
        throw;
    }
  }
}

// =================================================================================================

// Compiles the templated class
template class Xtrmv<half>;
template class Xtrmv<float>;
template class Xtrmv<double>;
template class Xtrmv<float2>;
template class Xtrmv<double2>;

// =================================================================================================
}  // namespace clblast
