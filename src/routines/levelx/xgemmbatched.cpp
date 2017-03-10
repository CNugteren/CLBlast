
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the XgemmBatched class (see the header for information about the class).
//
// =================================================================================================

#include "routines/levelx/xgemmbatched.hpp"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
XgemmBatched<T>::XgemmBatched(Queue &queue, EventPointer event, const std::string &name):
    Routine(queue, event, name,
            {"Copy","Pad","Transpose","Padtranspose","Xgemm","XgemmDirect","KernelSelection"},
            PrecisionValue<T>(), {}, {
    #include "../../kernels/level3/level3.opencl"
    #include "../../kernels/level3/copy_fast.opencl"
    #include "../../kernels/level3/copy_pad.opencl"
    #include "../../kernels/level3/transpose_fast.opencl"
    #include "../../kernels/level3/transpose_pad.opencl"
    #include "../../kernels/level3/convert_symmetric.opencl"
    #include "../../kernels/level3/convert_triangular.opencl"
    #include "../../kernels/level3/convert_hermitian.opencl"
    , // separated in multiple parts to prevent C1091 in MSVC 2013
    #include "../../kernels/level3/xgemm_direct_part1.opencl"
    #include "../../kernels/level3/xgemm_direct_part2.opencl"
    #include "../../kernels/level3/xgemm_direct_part3.opencl"
    , // separated in multiple parts to prevent C1091 in MSVC 2013
    #include "../../kernels/level3/xgemm_part1.opencl"
    #include "../../kernels/level3/xgemm_part2.opencl"
    #include "../../kernels/level3/xgemm_part3.opencl"
    }) {
}

// =================================================================================================

// The main routine
template <typename T>
void XgemmBatched<T>::DoGemmBatched(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                                    const size_t m, const size_t n, const size_t k,
                                    const std::vector<T> &alphas,
                                    const Buffer<T> & a_buffer, const std::vector<size_t> &a_offsets, const size_t a_ld,
                                    const Buffer<T> & b_buffer, const std::vector<size_t> &b_offsets, const size_t b_ld,
                                    const std::vector<T> &betas,
                                    const Buffer<T> & c_buffer, const std::vector<size_t> &c_offsets, const size_t c_ld,
                                    const size_t batch_count) {

  // Tests for a valid batch count
  if ((batch_count < 1) || (alphas.size() != batch_count) || (betas.size() != batch_count) ||
      (a_offsets.size() != batch_count) || (b_offsets.size() != batch_count) || (c_offsets.size() != batch_count)) {
    throw BLASError(StatusCode::kInvalidBatchCount);
  }

  // Makes sure all dimensions are larger than zero
  if ((m == 0) || (n == 0) || (k == 0)) { throw BLASError(StatusCode::kInvalidDimension); }

  // Computes whether or not the matrices are transposed in memory. See GEMM routine for details.
  const auto a_rotated = (layout == Layout::kColMajor && a_transpose != Transpose::kNo) ||
                         (layout == Layout::kRowMajor && a_transpose == Transpose::kNo);
  const auto b_rotated = (layout == Layout::kColMajor && b_transpose != Transpose::kNo) ||
                         (layout == Layout::kRowMajor && b_transpose == Transpose::kNo);
  const auto c_rotated = (layout == Layout::kRowMajor);
  static const auto a_want_rotated = false;
  static const auto b_want_rotated = true;
  static const auto c_want_rotated = false;
  const auto a_do_transpose = a_rotated != a_want_rotated;
  const auto b_do_transpose = b_rotated != b_want_rotated;
  const auto c_do_transpose = c_rotated != c_want_rotated;

  // In case of complex data-types, the transpose can also become a conjugate transpose
  const auto a_conjugate = (a_transpose == Transpose::kConjugate);
  const auto b_conjugate = (b_transpose == Transpose::kConjugate);

  // Computes the first and second dimensions of the 3 matrices taking into account whether the
  // matrices are rotated or not
  const auto a_one = (a_rotated) ? k : m;
  const auto a_two = (a_rotated) ? m : k;
  const auto b_one = (b_rotated) ? n : k;
  const auto b_two = (b_rotated) ? k : n;
  const auto c_one = (c_rotated) ? n : m;
  const auto c_two = (c_rotated) ? m : n;

  // Tests the matrices for validity
  for (auto batch = size_t{0}; batch < batch_count; ++batch) {
    TestMatrixA(a_one, a_two, a_buffer, a_offsets[batch], a_ld);
    TestMatrixB(b_one, b_two, b_buffer, b_offsets[batch], b_ld);
    TestMatrixC(c_one, c_two, c_buffer, c_offsets[batch], c_ld);
  }

  // StatusCode::kNotImplemented;
}

// =================================================================================================

// Compiles the templated class
template class XgemmBatched<half>;
template class XgemmBatched<float>;
template class XgemmBatched<double>;
template class XgemmBatched<float2>;
template class XgemmBatched<double2>;

// =================================================================================================
} // namespace clblast
