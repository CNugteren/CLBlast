
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
// and Jack Dongarra and the OpenCL implementation in clBLAS.
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

// The entry point: transforming into col-major (if needed) and then running the col-major version
template <typename T>
void Xtrsm<T>::DoTrsm(const Layout layout, Side side, Triangle triangle,
                      const Transpose a_transpose, const Diagonal diagonal,
                      size_t m, size_t n,
                      const T alpha,
                      const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                      const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld) {

  // Converts row-major to a col-major problem:
  // The idea is that
  //   B = A*X
  // can be computed as
  //   B' = (A*X)' = X'*A'
  // Since changing the order is basically a transpose on each matrix, the formula becomes:
  //   B = X*A
  // So only the side (left/right) and the triangle (upper/lower) are changed and M/N are swapped
  if (layout == Layout::kRowMajor) {
    std::swap(m, n);
    side = (side == Side::kLeft) ? Side::kRight : Side::kLeft;
    triangle = (triangle == Triangle::kLower) ? Triangle::kUpper : Triangle::kLower;
  }

  // Runs the col-major version of TRSM
  TrsmColMajor(side, triangle, a_transpose, diagonal,
               m, n, alpha,
               a_buffer, a_offset, a_ld,
               b_buffer, b_offset, b_ld);
}

// =================================================================================================

// The main routine
template <typename T>
void Xtrsm<T>::TrsmColMajor(const Side side, const Triangle triangle,
                            const Transpose a_transpose, const Diagonal diagonal,
                            const size_t m, const size_t n,
                            const T alpha,
                            const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                            const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld) {

  // Settings
  constexpr auto block_size = size_t{16}; // tuneable

  // Makes sure all dimensions are larger than zero
  if ((m == 0) || (n == 0)) { throw BLASError(StatusCode::kInvalidDimension); }

  // Some parts of this kernel are not tunable and thus require some minimal OpenCL properties
  if (device_.MaxWorkGroupSize() < 16) { // minimum of total local work size of 16
    throw RuntimeErrorCode(StatusCode::kNotImplemented);
  }

  // Computes the k dimension. This is based on whether or not matrix is A (on the left)
  // or B (on the right) in the Xgemm routine.
  const auto k = (side == Side::kLeft) ? m : n;

  // Checks for validity of the triangular A matrix
  TestMatrixA(k, k, a_buffer, a_offset, a_ld);

  // Checks for validity of the input B matrix
  TestMatrixB(m, n, b_buffer, b_offset, b_ld);

  // Creates a copy of B to avoid overwriting input in GEMM while computing output
  const auto b_size = b_ld * (n - 1) + m + b_offset;
  const auto x_one = m;
  const auto x_two = n;
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
  auto fill_matrix_event = Event();
  FillMatrix(queue_, device_, program_, fill_matrix_event.pointer(), eventWaitList,
             x_one, x_two, x_ld, x_offset, x_buffer, ConstantZero<T>(), 16);
  fill_matrix_event.WaitForCompletion();

  // Inverts the diagonal blocks
  auto diagonal_invert_event = Event();
  auto inverter = Xinvert<T>(queue_, diagonal_invert_event.pointer());
  inverter.InvertMatrixDiagonalBlocks(Layout::kColMajor, triangle, diagonal,
                                      k, block_size, a_buffer, a_offset, a_ld, a_inv_buffer);
  diagonal_invert_event.WaitForCompletion();

  // Derives properties based on the arguments
  const auto condition = ((triangle == Triangle::kUpper && a_transpose != Transpose::kNo) ||
                          (triangle == Triangle::kLower && a_transpose == Transpose::kNo));

  // Left side
  if (side == Side::kLeft) {

    // True when (lower triangular) or (upper triangular and transposed)
    if (condition) {
      for (auto i = size_t{0}; i < m; i += block_size) {
        const auto gemm_alpha = (i == 0) ? alpha : ConstantOne<T>();
        const auto current_block_size = std::min(m - i, block_size);
        auto gemm1_event = Event();
        auto gemm1 = Xgemm<T>(queue_, gemm1_event.pointer());
        gemm1.DoGemm(Layout::kColMajor, a_transpose, Transpose::kNo,
                     current_block_size, n, current_block_size, gemm_alpha,
                     a_inv_buffer, i * block_size, block_size,
                     b_buffer, b_offset + i, b_ld, ConstantZero<T>(),
                     x_buffer, x_offset + i, x_ld);
        gemm1_event.WaitForCompletion();
        if (i + block_size >= m) { break; }

        const auto this_a_offset = (a_transpose == Transpose::kNo) ? (i + block_size) + i * a_ld : i + (block_size + i) * a_ld;
        auto gemm2_event = Event();
        auto gemm2 = Xgemm<T>(queue_, gemm2_event.pointer());
        gemm2.DoGemm(Layout::kColMajor, a_transpose, Transpose::kNo,
                     m - i - block_size, n, block_size, ConstantNegOne<T>(),
                     a_buffer, this_a_offset + a_offset, a_ld,
                     x_buffer, x_offset + i, x_ld, gemm_alpha,
                     b_buffer, b_offset + i + block_size, b_ld);
        gemm2_event.WaitForCompletion();
      }
    }

    // True when (upper triangular) or (lower triangular and transposed)
    else {
      const auto special_block_size = (m % block_size == 0) ? block_size : (m % block_size);
      const auto i_start = static_cast<int>(m) - static_cast<int>(special_block_size);
      for (auto i = i_start; i >= 0; i -= static_cast<int>(block_size)) {
        const auto current_block_size = (i == i_start) ? special_block_size : block_size;
        const auto gemm_alpha = (i == i_start) ? alpha : ConstantOne<T>();
        auto gemm1_event = Event();
        auto gemm1 = Xgemm<T>(queue_, gemm1_event.pointer());
        gemm1.DoGemm(Layout::kColMajor, a_transpose, Transpose::kNo,
                     current_block_size, n, current_block_size, gemm_alpha,
                     a_inv_buffer, i * block_size, block_size,
                     b_buffer, b_offset + i, b_ld, ConstantZero<T>(),
                     x_buffer, x_offset + i, x_ld);
        gemm1_event.WaitForCompletion();
        if (i - static_cast<int>(block_size) < 0) { break; }

        const auto this_a_offset = (a_transpose == Transpose::kNo) ? i * a_ld : i;
        auto gemm2_event = Event();
        auto gemm2 = Xgemm<T>(queue_, gemm2_event.pointer());
        gemm2.DoGemm(Layout::kColMajor, a_transpose, Transpose::kNo,
                     i, n, current_block_size, ConstantNegOne<T>(),
                     a_buffer, this_a_offset + a_offset, a_ld,
                     x_buffer, x_offset + i, x_ld, gemm_alpha,
                     b_buffer, b_offset, b_ld);
        gemm2_event.WaitForCompletion();
      }
    }
  }

  // Right side
  else {

    // True when (lower triangular) or (upper triangular and transposed)
    if (condition) {
      const auto special_block_size = (n % block_size == 0) ? block_size : (n % block_size);
      const auto i_start = static_cast<int>(n) - static_cast<int>(special_block_size);
      for (auto i = i_start; i >= 0; i -= static_cast<int>(block_size)) {
        const auto current_block_size = (i == i_start) ? special_block_size : block_size;
        const auto gemm_alpha = (i == i_start) ? alpha : ConstantOne<T>();
        auto gemm1_event = Event();
        auto gemm1 = Xgemm<T>(queue_, gemm1_event.pointer());
        gemm1.DoGemm(Layout::kColMajor, Transpose::kNo, a_transpose,
                     m, current_block_size, current_block_size, gemm_alpha,
                     b_buffer, b_offset + i * b_ld, b_ld,
                     a_inv_buffer, i * block_size, block_size, ConstantZero<T>(),
                     x_buffer, x_offset + i * x_ld, x_ld);
        gemm1_event.WaitForCompletion();
        if (i - static_cast<int>(block_size) < 0) { break; }

        const auto this_a_offset = (a_transpose == Transpose::kNo) ? i : i * a_ld;
        auto gemm2_event = Event();
        auto gemm2 = Xgemm<T>(queue_, gemm2_event.pointer());
        gemm2.DoGemm(Layout::kColMajor, Transpose::kNo, a_transpose,
                     m, i, current_block_size, ConstantNegOne<T>(),
                     x_buffer, x_offset + i * x_ld, x_ld,
                     a_buffer, this_a_offset + a_offset, a_ld, gemm_alpha,
                     b_buffer, b_offset, b_ld);
        gemm2_event.WaitForCompletion();
      }
    }

    // True when (upper triangular) or (lower triangular and transposed)
    else {
      for (auto i = size_t{0}; i < n; i += block_size) {
        const auto gemm_alpha = (i == 0) ? alpha : ConstantOne<T>();
        const auto current_block_size = std::min(n - i, block_size);
        auto gemm1_event = Event();
        auto gemm1 = Xgemm<T>(queue_, gemm1_event.pointer());
        gemm1.DoGemm(Layout::kColMajor, Transpose::kNo, a_transpose,
                     m, current_block_size, current_block_size, gemm_alpha,
                     b_buffer, b_offset + i * b_ld, b_ld,
                     a_inv_buffer, i * block_size, block_size, ConstantZero<T>(),
                     x_buffer, x_offset + i * x_ld, x_ld);
        gemm1_event.WaitForCompletion();
        if (i + block_size >= n) { break; }

        const auto this_a_offset = (a_transpose == Transpose::kNo) ? i + (block_size + i) * a_ld : (i + block_size) + i * a_ld;
        auto gemm2_event = Event();
        auto gemm2 = Xgemm<T>(queue_, gemm2_event.pointer());
        gemm2.DoGemm(Layout::kColMajor, Transpose::kNo, a_transpose,
                     m, n - i - block_size, block_size, ConstantNegOne<T>(),
                     x_buffer, x_offset + i * x_ld, x_ld,
                     a_buffer, this_a_offset + a_offset, a_ld, gemm_alpha,
                     b_buffer, b_offset + (i + block_size) * b_ld, b_ld);
        gemm2_event.WaitForCompletion();
      }
    }
  }

  // Retrieves the results
  x_buffer.CopyToAsync(queue_, b_size, b_buffer, event_);
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
