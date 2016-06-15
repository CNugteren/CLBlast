
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

#include "internal/routines/level3/xgemm.h"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Specific implementations to get the memory-type based on a template argument
template <> const Precision Xgemm<half>::precision_ = Precision::kHalf;
template <> const Precision Xgemm<float>::precision_ = Precision::kSingle;
template <> const Precision Xgemm<double>::precision_ = Precision::kDouble;
template <> const Precision Xgemm<float2>::precision_ = Precision::kComplexSingle;
template <> const Precision Xgemm<double2>::precision_ = Precision::kComplexDouble;

// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xgemm<T>::Xgemm(Queue &queue, EventPointer event, const std::string &name):
    Routine<T>(queue, event, name, {"Copy","Pad","Transpose","Padtranspose","Xgemm"}, precision_) {
  source_string_ =
    #include "../../kernels/level3/level3.opencl"
    #include "../../kernels/level3/copy_fast.opencl"
    #include "../../kernels/level3/copy_pad.opencl"
    #include "../../kernels/level3/transpose_fast.opencl"
    #include "../../kernels/level3/transpose_pad.opencl"
    #include "../../kernels/level3/convert_symmetric.opencl"
    #include "../../kernels/level3/convert_triangular.opencl"
    #include "../../kernels/level3/convert_hermitian.opencl"
    #include "../../kernels/level3/xgemm_part1.opencl"
    #include "../../kernels/level3/xgemm_part2.opencl"
  ;
}

// =================================================================================================

// The main routine
template <typename T>
StatusCode Xgemm<T>::DoGemm(const Layout layout,
                            const Transpose a_transpose, const Transpose b_transpose,
                            const size_t m, const size_t n, const size_t k,
                            const T alpha,
                            const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                            const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld,
                            const T beta,
                            const Buffer<T> &c_buffer, const size_t c_offset, const size_t c_ld) {

  // Makes sure all dimensions are larger than zero
  if ((m == 0) || (n == 0) || (k == 0)) { return StatusCode::kInvalidDimension; }

  // Computes whether or not the matrices are transposed in memory. This is based on their layout
  // (row or column-major) and whether or not they are requested to be pre-transposed. Note
  // that the Xgemm kernel expects either matrices A and C (in case of row-major) or B (in case of
  // col-major) to be transformed, so transposing requirements are not the same as whether or not
  // the matrix is actually transposed in memory.
  const auto a_rotated = (layout == Layout::kColMajor && a_transpose != Transpose::kNo) ||
                         (layout == Layout::kRowMajor && a_transpose == Transpose::kNo);
  const auto b_rotated = (layout == Layout::kColMajor && b_transpose != Transpose::kNo) ||
                         (layout == Layout::kRowMajor && b_transpose == Transpose::kNo);
  const auto c_rotated = (layout == Layout::kRowMajor);
  const auto a_do_transpose =  a_rotated;
  const auto b_do_transpose = !b_rotated;
  const auto c_do_transpose =  c_rotated;

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
  const auto m_ceiled = Ceil(m, db_["MWG"]);
  const auto n_ceiled = Ceil(n, db_["NWG"]);
  const auto k_ceiled = Ceil(k, db_["KWG"]);

  // The padded/transposed input/output matrices: if memory allocation fails, throw an exception
  try {

    // Loads the program from the database
    const auto program = GetProgramFromCache();

    // Determines whether or not temporary matrices are needed
    auto a_no_temp = a_one == m_ceiled && a_two == k_ceiled && a_ld == m_ceiled && a_offset == 0 &&
                     a_do_transpose == false && a_conjugate == false;
    auto b_no_temp = b_one == n_ceiled && b_two == k_ceiled && b_ld == n_ceiled && b_offset == 0 &&
                     b_do_transpose == false && b_conjugate == false;
    auto c_no_temp = c_one == m_ceiled && c_two == n_ceiled && c_ld == m_ceiled && c_offset == 0 &&
                     c_do_transpose == false;

    // Creates the temporary matrices
    const auto a_temp = (a_no_temp) ? a_buffer : Buffer<T>(context_, k_ceiled*m_ceiled);
    const auto b_temp = (b_no_temp) ? b_buffer : Buffer<T>(context_, k_ceiled*n_ceiled);
    const auto c_temp = (c_no_temp) ? c_buffer : Buffer<T>(context_, m_ceiled*n_ceiled);

    // Upload the scalar arguments as constant buffers to the device (needed for half-precision)
    auto alpha_buffer = Buffer<T>(context_, 1);
    auto beta_buffer = Buffer<T>(context_, 1);
    alpha_buffer.Write(queue_, 1, &alpha);
    beta_buffer.Write(queue_, 1, &beta);

    // Events of all kernels (including pre/post processing kernels)
    auto eventWaitList = std::vector<Event>();
    auto emptyEventList = std::vector<Event>();

    // Runs the pre-processing kernel for matrix A. This transposes the matrix, but also pads zeros
    // to fill it up until it reaches a certain multiple of size (kernel parameter dependent). In
    // case nothing has to be done, these kernels can be skipped.
    if (!a_no_temp) {
      auto eventProcessA = Event();
      status = PadCopyTransposeMatrix(eventProcessA.pointer(), emptyEventList,
                                      a_one, a_two, a_ld, a_offset, a_buffer,
                                      m_ceiled, k_ceiled, m_ceiled, 0, a_temp,
                                      program, true, a_do_transpose, a_conjugate);
      if (ErrorIn(status)) { return status; }
      eventWaitList.push_back(eventProcessA);
    }

    // As above, but now for matrix B
    if (!b_no_temp) {
      auto eventProcessB = Event();
      status = PadCopyTransposeMatrix(eventProcessB.pointer(), emptyEventList,
                                      b_one, b_two, b_ld, b_offset, b_buffer,
                                      n_ceiled, k_ceiled, n_ceiled, 0, b_temp,
                                      program, true, b_do_transpose, b_conjugate);
      if (ErrorIn(status)) { return status; }
      eventWaitList.push_back(eventProcessB);
    }

    // As above, but now for matrix C. This is only necessary if C is used both as input and output.
    if (!c_no_temp && beta != static_cast<T>(0)) {
      auto eventProcessC = Event();
      status = PadCopyTransposeMatrix(eventProcessC.pointer(), emptyEventList,
                                      c_one, c_two, c_ld, c_offset, c_buffer,
                                      m_ceiled, n_ceiled, m_ceiled, 0, c_temp,
                                      program, true, c_do_transpose, false);
      if (ErrorIn(status)) { return status; }
      eventWaitList.push_back(eventProcessC);
    }

    // Retrieves the Xgemm kernel from the compiled binary
    try {
      auto kernel = Kernel(program, "Xgemm");

      // Sets the kernel arguments
      kernel.SetArgument(0, static_cast<int>(m_ceiled));
      kernel.SetArgument(1, static_cast<int>(n_ceiled));
      kernel.SetArgument(2, static_cast<int>(k_ceiled));
      kernel.SetArgument(3, alpha_buffer());
      kernel.SetArgument(4, beta_buffer());
      kernel.SetArgument(5, a_temp());
      kernel.SetArgument(6, b_temp());
      kernel.SetArgument(7, c_temp());

      // Computes the global and local thread sizes
      const auto global = std::vector<size_t>{
        (m_ceiled * db_["MDIMC"]) / db_["MWG"],
        (n_ceiled * db_["NDIMC"]) / db_["NWG"]
      };
      const auto local = std::vector<size_t>{db_["MDIMC"], db_["NDIMC"]};

      // Launches the kernel
      auto eventKernel = Event();
      auto eventPointer = (!c_no_temp) ? eventKernel.pointer() : event_;
      status = RunKernel(kernel, global, local, eventPointer, eventWaitList);
      if (ErrorIn(status)) { return status; }

      // Runs the post-processing kernel if needed
      if (!c_no_temp) {
        eventWaitList.push_back(eventKernel);
        status = PadCopyTransposeMatrix(event_, eventWaitList,
                                        m_ceiled, n_ceiled, m_ceiled, 0, c_temp,
                                        c_one, c_two, c_ld, c_offset, c_buffer,
                                        program, false, c_do_transpose, false);
        if (ErrorIn(status)) { return status; }
      }

      // Successfully finished the computation
      return StatusCode::kSuccess;
    } catch (...) { return StatusCode::kInvalidKernel; }
  } catch (...) { return StatusCode::kTempBufferAllocFailure; }
}

// =================================================================================================

// Compiles the templated class
template class Xgemm<half>;
template class Xgemm<float>;
template class Xgemm<double>;
template class Xgemm<float2>;
template class Xgemm<double2>;

// =================================================================================================
} // namespace clblast
