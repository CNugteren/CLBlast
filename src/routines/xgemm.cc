
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xgemm class (see the header for information about the class).
//
// =================================================================================================

#include "internal/routines/xgemm.h"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Specific implementations to get the memory-type based on a template argument
template <> const Precision Xgemm<float>::precision_ = Precision::kSingle;
template <> const Precision Xgemm<double>::precision_ = Precision::kDouble;
template <> const Precision Xgemm<float2>::precision_ = Precision::kComplexSingle;
template <> const Precision Xgemm<double2>::precision_ = Precision::kComplexDouble;

// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xgemm<T>::Xgemm(CommandQueue &queue, Event &event):
    Routine(queue, event, {"Copy", "Pad", "Transpose", "PadTranspose", "Xgemm"}, precision_) {
}

// =================================================================================================

// The main routine
template <typename T>
StatusCode Xgemm<T>::DoGemm(const Layout layout,
                            const Transpose a_transpose, const Transpose b_transpose,
                            const size_t m, const size_t n, const size_t k,
                            const T alpha,
                            const Buffer &a_buffer, const size_t a_offset, const size_t a_ld,
                            const Buffer &b_buffer, const size_t b_offset, const size_t b_ld,
                            const T beta,
                            const Buffer &c_buffer, const size_t c_offset, const size_t c_ld) {

  // Makes sure all dimensions are larger than zero
  if ((m == 0) || (n == 0) || (k == 0)) { return StatusCode::kInvalidDimension; }

  // Computes whether or not the matrices are transposed in memory. This is based on their layout
  // (row or column-major) and whether or not they are requested to be pre-transposed. Note
  // that the Xgemm kernel expects either matrices A and C (in case of row-major) or B (in case of
  // col-major) to be transformed, so transposing requirements are not the same as whether or not
  // the matrix is actually transposed in memory.
  auto a_rotated = (layout == Layout::kColMajor && a_transpose != Transpose::kNo) ||
                   (layout == Layout::kRowMajor && a_transpose == Transpose::kNo);
  auto b_rotated = (layout == Layout::kColMajor && b_transpose != Transpose::kNo) ||
                   (layout == Layout::kRowMajor && b_transpose == Transpose::kNo);
  auto c_rotated = (layout == Layout::kRowMajor);
  auto a_do_transpose =  a_rotated;
  auto b_do_transpose = !b_rotated;
  auto c_do_transpose =  c_rotated;

  // In case of complex data-types, the transpose can also become a conjugate transpose
  auto a_conjugate = (a_transpose == Transpose::kConjugate);
  auto b_conjugate = (b_transpose == Transpose::kConjugate);

  // Computes the first and second dimensions of the 3 matrices taking into account whether the
  // matrices are rotated or not
  auto a_one = (a_rotated) ? k : m;
  auto a_two = (a_rotated) ? m : k;
  auto b_one = (b_rotated) ? n : k;
  auto b_two = (b_rotated) ? k : n;
  auto c_one = (c_rotated) ? n : m;
  auto c_two = (c_rotated) ? m : n;

  // Tests three matrices (A, B, C) for validity, first from a perspective of the OpenCL buffers and
  // their sizes, and then from a perspective of parameter values (e.g. m, n, k). Tests whether the
  // OpenCL buffers are valid and non-zero and whether the OpenCL buffers have sufficient storage
  // space. Also tests that the leading dimensions of:
  //    matrix A cannot be less than K when rotated, or less than M when not-rotated
  //    matrix B cannot be less than N when rotated, or less than K when not-rotated
  //    matrix C cannot be less than N when rotated, or less than M when not-rotated
  auto status = TestMatrixA(a_one, a_two, a_buffer, a_offset, a_ld, sizeof(T));
  if (ErrorIn(status)) { return status; }
  status = TestMatrixB(b_one, b_two, b_buffer, b_offset, b_ld, sizeof(T));
  if (ErrorIn(status)) { return status; }
  status = TestMatrixC(c_one, c_two, c_buffer, c_offset, c_ld, sizeof(T));
  if (ErrorIn(status)) { return status; }

  // Calculates the ceiled versions of m, n, and k
  auto m_ceiled = Ceil(m, db_["MWG"]);
  auto n_ceiled = Ceil(n, db_["NWG"]);
  auto k_ceiled = Ceil(k, db_["KWG"]);

  // Allocates space on the device for padded and/or transposed input and output matrices.
  try {
    auto temp_a = Buffer(context_, CL_MEM_READ_WRITE, k_ceiled*m_ceiled*sizeof(T));
    auto temp_b = Buffer(context_, CL_MEM_READ_WRITE, k_ceiled*n_ceiled*sizeof(T));
    auto temp_c = Buffer(context_, CL_MEM_READ_WRITE, m_ceiled*n_ceiled*sizeof(T));

    // Loads the program from the database
    auto& program = GetProgramFromCache();

    // Runs the pre-processing kernels. This transposes the matrices, but also pads zeros to fill
    // them up until they reach a certain multiple of size (kernel parameter dependent).
    status = PadCopyTransposeMatrix(a_one, a_two, a_ld, a_offset, a_buffer,
                                    m_ceiled, k_ceiled, m_ceiled, 0, temp_a,
                                    a_do_transpose, a_conjugate, true, program);
    if (ErrorIn(status)) { return status; }
    status = PadCopyTransposeMatrix(b_one, b_two, b_ld, b_offset, b_buffer,
                                    n_ceiled, k_ceiled, n_ceiled, 0, temp_b,
                                    b_do_transpose, b_conjugate, true, program);
    if (ErrorIn(status)) { return status; }

    // Only necessary for matrix C if it used both as input and output
    if (beta != static_cast<T>(0)) {
      status = PadCopyTransposeMatrix(c_one, c_two, c_ld, c_offset, c_buffer,
                                      m_ceiled, n_ceiled, m_ceiled, 0, temp_c,
                                      c_do_transpose, false, true, program);
      if (ErrorIn(status)) { return status; }
    }

    // Retrieves the Xgemm kernel from the compiled binary
    try {
      auto kernel = Kernel(program, "Xgemm");

      // Sets the kernel arguments
      kernel.SetArgument(0, static_cast<int>(m_ceiled));
      kernel.SetArgument(1, static_cast<int>(n_ceiled));
      kernel.SetArgument(2, static_cast<int>(k_ceiled));
      kernel.SetArgument(3, alpha);
      kernel.SetArgument(4, beta);
      kernel.SetArgument(5, temp_a());
      kernel.SetArgument(6, temp_b());
      kernel.SetArgument(7, temp_c());

      // Computes the global and local thread sizes
      auto global = std::vector<size_t>{
        (m_ceiled * db_["MDIMC"]) / db_["MWG"],
        (n_ceiled * db_["NDIMC"]) / db_["NWG"]
      };
      auto local = std::vector<size_t>{db_["MDIMC"], db_["NDIMC"]};

      // Launches the kernel
      status = RunKernel(kernel, global, local);
      if (ErrorIn(status)) { return status; }

      // Runs the post-processing kernel
      status = PadCopyTransposeMatrix(m_ceiled, n_ceiled, m_ceiled, 0, temp_c,
                                      c_one, c_two, c_ld, c_offset, c_buffer,
                                      c_do_transpose, false, false, program);
      if (ErrorIn(status)) { return status; }

      // Successfully finished the computation
      return StatusCode::kSuccess;
    } catch (...) { return StatusCode::kInvalidKernel; }
  } catch (...) { return StatusCode::kTempBufferAllocFailure; }
}

// =================================================================================================

// Compiles the templated class
template class Xgemm<float>;
template class Xgemm<double>;
template class Xgemm<float2>;
template class Xgemm<double2>;

// =================================================================================================
} // namespace clblast
