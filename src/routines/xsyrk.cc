
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xsyrk class (see the header for information about the class).
//
// =================================================================================================

#include "internal/routines/xsyrk.h"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Specific implementations to get the memory-type based on a template argument
template <> const Precision Xsyrk<float>::precision_ = Precision::kSingle;
template <> const Precision Xsyrk<double>::precision_ = Precision::kDouble;
template <> const Precision Xsyrk<float2>::precision_ = Precision::kComplexSingle;
template <> const Precision Xsyrk<double2>::precision_ = Precision::kComplexDouble;

// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xsyrk<T>::Xsyrk(CommandQueue &queue, Event &event):
    Routine(queue, event, {"Copy", "Pad", "Transpose", "PadTranspose", "Xgemm"}, precision_) {
}

// =================================================================================================

// The main routine
template <typename T>
StatusCode Xsyrk<T>::DoSyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                            const size_t n, const size_t k,
                            const T alpha,
                            const Buffer &a_buffer, const size_t a_offset, const size_t a_ld,
                            const T beta,
                            const Buffer &c_buffer, const size_t c_offset, const size_t c_ld) {

  // Makes sure all dimensions are larger than zero
  if ((n == 0) || (k == 0) ) { return StatusCode::kInvalidDimension; }

  // Computes whether or not the matrices are transposed in memory. This is based on their layout
  // (row or column-major) and whether or not they are requested to be pre-transposed.
  auto a_rotated = (layout == Layout::kColMajor && a_transpose != Transpose::kNo) ||
                   (layout == Layout::kRowMajor && a_transpose == Transpose::kNo);
  auto c_rotated = (layout == Layout::kRowMajor);

  // In case of complex data-types, the transpose can also become a conjugate transpose
  auto a_conjugate = (a_transpose == Transpose::kConjugate);

  // Computes the first and second dimensions of the A matrix taking the layout into account
  auto a_one = (a_rotated) ? k : n;
  auto a_two = (a_rotated) ? n : k;

  // Tests the two matrices (A, C) for validity, first from a perspective of the OpenCL buffers and
  // their sizes, and then from a perspective of parameter values (e.g. n, k). Tests whether the
  // OpenCL buffers are valid and non-zero and whether the OpenCL buffers have sufficient storage
  // space. Also tests that the leading dimensions of:
  //    matrix A cannot be less than N when rotated, or less than K when not-rotated
  //    matrix C cannot be less than N
  auto status = TestMatrixA(a_one, a_two, a_buffer, a_offset, a_ld, sizeof(T));
  if (ErrorIn(status)) { return status; }
  status = TestMatrixC(n, n, c_buffer, c_offset, c_ld, sizeof(T));
  if (ErrorIn(status)) { return status; }

  // Calculates the ceiled versions of n and k
  auto n_ceiled = Ceil(n, db_["NWG"]);
  auto k_ceiled = Ceil(k, db_["KWG"]);

  // Decides which kernel to run: the upper-triangular or lower-triangular version
  auto kernel_name = (triangle == Triangle::kUpper) ? "XgemmUpper" : "XgemmLower";

  // Allocates space on the device for padded and/or transposed input and output matrices.
  try {
    auto temp_a = Buffer(context_, CL_MEM_READ_WRITE, k_ceiled*n_ceiled*sizeof(T));
    auto temp_c = Buffer(context_, CL_MEM_READ_WRITE, n_ceiled*n_ceiled*sizeof(T));

    // Loads the program from the database
    auto& program = GetProgramFromCache();

    // Runs the pre-processing kernel. This transposes the matrix A, but also pads zeros to
    // fill them up until they reach a certain multiple of size (kernel parameter dependent).
    status = PadCopyTransposeMatrix(a_one, a_two, a_ld, a_offset, a_buffer,
                                    n_ceiled, k_ceiled, n_ceiled, 0, temp_a,
                                    a_rotated, a_conjugate, true, false, false, program);
    if (ErrorIn(status)) { return status; }

    // Furthermore, also creates a (possibly padded) copy of matrix C, since it is not allowed to
    // modify the other triangle.
    status = PadCopyTransposeMatrix(n, n, c_ld, c_offset, c_buffer,
                                    n_ceiled, n_ceiled, n_ceiled, 0, temp_c,
                                    c_rotated, false, true, false, false, program);
    if (ErrorIn(status)) { return status; }

    // Retrieves the XgemmUpper or XgemmLower kernel from the compiled binary
    try {
      auto kernel = Kernel(program, kernel_name);

      // Sets the kernel arguments
      kernel.SetArgument(0, static_cast<int>(n_ceiled));
      kernel.SetArgument(1, static_cast<int>(k_ceiled));
      kernel.SetArgument(2, alpha);
      kernel.SetArgument(3, beta);
      kernel.SetArgument(4, temp_a());
      kernel.SetArgument(5, temp_a());
      kernel.SetArgument(6, temp_c());

      // Computes the global and local thread sizes
      auto global = std::vector<size_t>{
        (n_ceiled * db_["MDIMC"]) / db_["MWG"],
        (n_ceiled * db_["NDIMC"]) / db_["NWG"]
      };
      auto local = std::vector<size_t>{db_["MDIMC"], db_["NDIMC"]};

      // Launches the kernel
      status = RunKernel(kernel, global, local);
      if (ErrorIn(status)) { return status; }

      // Runs the post-processing kernel
      auto upper = (triangle == Triangle::kUpper);
      auto lower = (triangle == Triangle::kLower);
      status = PadCopyTransposeMatrix(n_ceiled, n_ceiled, n_ceiled, 0, temp_c,
                                      n, n, c_ld, c_offset, c_buffer,
                                      c_rotated, false, false, upper, lower, program);
      if (ErrorIn(status)) { return status; }

      // Successfully finished the computation
      return StatusCode::kSuccess;
    } catch (...) { return StatusCode::kInvalidKernel; }
  } catch (...) { return StatusCode::kTempBufferAllocFailure; }
}

// =================================================================================================

// Compiles the templated class
template class Xsyrk<float>;
template class Xsyrk<double>;
template class Xsyrk<float2>;
template class Xsyrk<double2>;

// =================================================================================================
} // namespace clblast
