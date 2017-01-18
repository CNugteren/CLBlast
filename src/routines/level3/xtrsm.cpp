
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the triangular matrix solver (A * X = B) TRSM class. This code is based
// on the TRSM implementation in the CUDA version of Magma version 2.2.0 and the poster "Triangular
// Linear System Solver for GPU with CUDA and OpenCL" by Peng Du, Stanimire Tomov, Piotr Luszczek,
// and Jack Dongarra.
//
// =================================================================================================

#include "routines/level3/xtrsm.hpp"
#include "routines/levelx/xinvert.hpp"

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

  // Settings
  constexpr auto block_size = size_t{32}; // tuneable

  // Makes sure all dimensions are larger than zero
  if ((m == 0) || (n == 0)) { throw BLASError(StatusCode::kInvalidDimension); }

  // Computes the k dimension. This is based on whether or not matrix is A (on the left)
  // or B (on the right) in the Xgemm routine.
  const auto k = (side == Side::kLeft) ? m : n;

  // Checks for validity of the triangular A matrix
  TestMatrixA(k, k, a_buffer, a_offset, a_ld);

  // Determines which kernels to run based on the layout (the kernels assume column-major as
  // default) and on whether we are dealing with an upper or lower triangle of the triangular matrix
  const bool is_upper = ((triangle == Triangle::kUpper && layout != Layout::kRowMajor) ||
                         (triangle == Triangle::kLower && layout == Layout::kRowMajor));

  // Checks for validity of the input B matrix
  const auto b_one = (layout == Layout::kRowMajor) ? n : m;
  const auto b_two = (layout == Layout::kRowMajor) ? m : n;
  TestMatrixB(b_one, b_two, b_buffer, b_offset, b_ld);

  // Creates a copy of B to avoid overwriting input in GEMM while computing output
  const auto b_size = b_ld * (b_two - 1) + b_one + b_offset;
  const auto x_one = b_one;
  const auto x_size = b_size;
  const auto x_ld = b_ld;
  const auto x_offset = b_offset;
  auto x_buffer = Buffer<T>(context_, x_size);
  b_buffer.CopyTo(queue_, x_size, x_buffer);

  // Temporary buffer for the inverse of the A matrix
  const auto a_inv_size = Ceil(k, block_size) * block_size;
  auto a_inv_buffer = Buffer<T>(context_, a_inv_size);

  // Fills the output buffer with zeros
  auto eventWaitList = std::vector<Event>();
  const auto program = GetProgramFromCache(context_, PrecisionValue<T>(), "TRSM");
  auto fill_matrix_event = Event();
  FillMatrix(queue_, device_, program, db_, fill_matrix_event.pointer(), eventWaitList,
             x_one, x_ld, x_offset, x_buffer, ConstantZero<T>());
  fill_matrix_event.WaitForCompletion();

  // Inverts the diagonal blocks
  auto diagonal_invert_event = Event();
  auto inverter = Xinvert<T>(queue_, diagonal_invert_event.pointer());
  inverter.InvertMatrixDiagonalBlocks(layout, triangle, diagonal,
                                      k, block_size, a_buffer, a_offset, a_ld, a_inv_buffer);
  diagonal_invert_event.WaitForCompletion();

  // Lower of upper triangular
  const bool condition = ((triangle == Triangle::kUpper && a_transpose != Transpose::kNo) ||
                          (triangle == Triangle::kLower && a_transpose == Transpose::kNo));

  // Left side
  if (side == Side::kLeft) {

    // True when (lower triangular) or (upper triangular and transposed)
    if (condition) {
      for (auto i = size_t{0}; i < m; i += block_size) {
        const auto gemm_alpha = (i == 0) ? alpha : ConstantOne<T>();
        const auto current_block_size = std::min(m - i, block_size);
        DoGemm(layout, a_transpose, Transpose::kNo,
               current_block_size, n, current_block_size, gemm_alpha,
               a_inv_buffer, i * block_size, block_size,
               b_buffer, i, b_ld, ConstantZero<T>(),
               x_buffer, i, x_ld);
        if (i + block_size >= m) { break; }
        const auto this_a_offset = (a_transpose == Transpose::kNo) ? (i + block_size) + i * a_ld : i + (block_size + i) * a_ld;
        DoGemm(layout, a_transpose, Transpose::kNo,
               m - i - block_size, n, block_size, ConstantNegOne<T>(),
               a_buffer, this_a_offset, a_ld,
               x_buffer, i, x_ld, ConstantOne<T>(),
               b_buffer, i + block_size, b_ld);
      }
    }

    // True when (upper triangular) or (lower triangular and transposed)
    else {
      const auto current_block_size = (m % block_size == 0) ? block_size : (m % block_size);
      const auto i_start = static_cast<int>(m) - static_cast<int>(current_block_size);
      for (auto i = i_start; i >= 0; i -= static_cast<int>(block_size)) {
        const auto gemm_alpha = (i == i_start) ? alpha : ConstantOne<T>();
        DoGemm(layout, a_transpose, Transpose::kNo,
               block_size, n, block_size, gemm_alpha,
               a_inv_buffer, i * block_size, block_size,
               b_buffer, i, b_ld, ConstantZero<T>(),
               x_buffer, i, x_ld);
        if (i - static_cast<int>(block_size) < 0) { break; }
        const auto this_a_offset = (a_transpose == Transpose::kNo) ? i * a_ld : i;
        DoGemm(layout, a_transpose, Transpose::kNo,
               i, n, block_size, ConstantNegOne<T>(),
               a_buffer, this_a_offset, a_ld,
               x_buffer, i, x_ld, ConstantOne<T>(),
               b_buffer, 0, b_ld);
      }
    }
  }

  // Right side
  else {

    // True when (lower triangular) or (upper triangular and transposed)
    if (condition) {
      const auto current_block_size = (n % block_size == 0) ? block_size : (n % block_size);
      const auto i_start = static_cast<int>(n) - static_cast<int>(current_block_size);
      for (auto i = i_start; i >= 0; i -= static_cast<int>(block_size)) {
        const auto gemm_alpha = (i == i_start) ? alpha : ConstantOne<T>();
        DoGemm(layout, Transpose::kNo, a_transpose,
               m, block_size, block_size, gemm_alpha,
               b_buffer, i * b_ld, b_ld,
               a_inv_buffer, i * block_size, block_size, ConstantZero<T>(),
               x_buffer, i * x_ld, x_ld);
        if (i - static_cast<int>(block_size) < 0) { break; }
        const auto this_a_offset = (a_transpose == Transpose::kNo) ? i : i * a_ld;
        DoGemm(layout, Transpose::kNo, a_transpose,
               m, i, block_size, ConstantNegOne<T>(),
               x_buffer, i * x_ld, x_ld,
               a_buffer, this_a_offset, a_ld, ConstantOne<T>(),
               b_buffer, 0, b_ld);
      }
    }

    // True when (upper triangular) or (lower triangular and transposed)
    else {
      for (auto i = size_t{0}; i < n; i += block_size) {
        const auto gemm_alpha = (i == 0) ? alpha : ConstantOne<T>();
        const auto current_block_size = std::min(n - i, block_size);
        DoGemm(layout, Transpose::kNo, a_transpose,
               m, current_block_size, current_block_size, gemm_alpha,
               b_buffer, i * b_ld, b_ld,
               a_inv_buffer, i * block_size, block_size, ConstantZero<T>(),
               x_buffer, i * x_ld, x_ld);
        if (i + block_size >= n) { break; }
        const auto this_a_offset = (a_transpose == Transpose::kNo) ? i + (block_size + i) * a_ld : (i + block_size) + i * a_ld;
        DoGemm(layout, Transpose::kNo, a_transpose,
               m, n - i - block_size, block_size, ConstantNegOne<T>(),
               x_buffer, i * x_ld, x_ld,
               a_buffer, this_a_offset, a_ld, ConstantOne<T>(),
               b_buffer, (i + block_size) * b_ld, b_ld);
      }
    }
  }

  // Retrieves the results
  x_buffer.CopyTo(queue_, b_size, b_buffer);
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
