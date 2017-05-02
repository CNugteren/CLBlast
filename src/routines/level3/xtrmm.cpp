
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xtrmm class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level3/xtrmm.hpp"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xtrmm<T>::Xtrmm(Queue &queue, EventPointer event, const std::string &name):
    Xgemm<T>(queue, event, name) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xtrmm<T>::DoTrmm(const Layout layout, const Side side, const Triangle triangle,
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

  // Determines which kernel to run based on the layout (the Xgemm kernel assumes column-major as
  // default) and on whether we are dealing with an upper or lower triangle of the triangular matrix
  bool is_upper = ((triangle == Triangle::kUpper && layout != Layout::kRowMajor) ||
                   (triangle == Triangle::kLower && layout == Layout::kRowMajor));
  auto kernel_name = (is_upper) ? "TriaUpperToSquared" : "TriaLowerToSquared";

  // Determines whether or not the triangular matrix is unit-diagonal
  auto unit_diagonal = (diagonal == Diagonal::kUnit) ? true : false;

  // Temporary buffer for a copy of the triangular matrix
  auto temp_triangular = Buffer<T>(context_, k*k);

  // Creates a general matrix from the triangular matrix to be able to run the regular Xgemm
  // routine afterwards
  auto kernel = Kernel(program_, kernel_name);

  // Sets the arguments for the triangular-to-squared kernel
  kernel.SetArgument(0, static_cast<int>(k));
  kernel.SetArgument(1, static_cast<int>(a_ld));
  kernel.SetArgument(2, static_cast<int>(a_offset));
  kernel.SetArgument(3, a_buffer());
  kernel.SetArgument(4, static_cast<int>(k));
  kernel.SetArgument(5, static_cast<int>(k));
  kernel.SetArgument(6, static_cast<int>(0));
  kernel.SetArgument(7, temp_triangular());
  kernel.SetArgument(8, static_cast<int>(unit_diagonal));

  // Uses the common padding kernel's thread configuration. This is allowed, since the
  // triangular-to-squared kernel uses the same parameters.
  auto global = std::vector<size_t>{Ceil(CeilDiv(k, db_["PAD_WPTX"]), db_["PAD_DIMX"]),
                                    Ceil(CeilDiv(k, db_["PAD_WPTY"]), db_["PAD_DIMY"])};
  auto local = std::vector<size_t>{db_["PAD_DIMX"], db_["PAD_DIMY"]};
  auto kernelEvent = Event();
  RunKernel(kernel, queue_, device_, global, local, kernelEvent.pointer());

  // Synchronize now: 'DoGemm' does not accept a list of events to wait for
  kernelEvent.WaitForCompletion();

  // Runs the regular Xgemm code with either "B := alpha*A*B" or ...
  if (side == Side::kLeft) {
    DoGemm(layout, a_transpose, Transpose::kNo,
           m, n, k,
           alpha,
           temp_triangular, 0, k,
           b_buffer_copy, b_offset, b_ld,
           ConstantZero<T>(),
           b_buffer, b_offset, b_ld);
  }

  // ... with "B := alpha*B*A". Note that A and B are now reversed.
  else {
    try {
      DoGemm(layout, Transpose::kNo, a_transpose,
             m, n, k,
             alpha,
             b_buffer_copy, b_offset, b_ld,
             temp_triangular, 0, k,
             ConstantZero<T>(),
             b_buffer, b_offset, b_ld);
    } catch (BLASError &e) {
      // A and B are now reversed, so also reverse the error codes returned from the Xgemm routine
      switch(e.status()) {
        case StatusCode::kInvalidMatrixA:      throw BLASError(StatusCode::kInvalidMatrixB, e.details());
        case StatusCode::kInvalidMatrixB:      throw BLASError(StatusCode::kInvalidMatrixA, e.details());
        case StatusCode::kInvalidLeadDimA:     throw BLASError(StatusCode::kInvalidLeadDimB, e.details());
        case StatusCode::kInvalidLeadDimB:     throw BLASError(StatusCode::kInvalidLeadDimA, e.details());
        case StatusCode::kInsufficientMemoryA: throw BLASError(StatusCode::kInsufficientMemoryB, e.details());
        case StatusCode::kInsufficientMemoryB: throw BLASError(StatusCode::kInsufficientMemoryA, e.details());
        default:                               throw;
      }
    }
  }
}

// =================================================================================================

// Compiles the templated class
template class Xtrmm<half>;
template class Xtrmm<float>;
template class Xtrmm<double>;
template class Xtrmm<float2>;
template class Xtrmm<double2>;

// =================================================================================================
} // namespace clblast
