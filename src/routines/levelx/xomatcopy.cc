
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xomatcopy class (see the header for information about the class).
//
// =================================================================================================

#include "internal/routines/levelx/xomatcopy.h"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Specific implementations to get the memory-type based on a template argument
template <> const Precision Xomatcopy<half>::precision_ = Precision::kHalf;
template <> const Precision Xomatcopy<float>::precision_ = Precision::kSingle;
template <> const Precision Xomatcopy<double>::precision_ = Precision::kDouble;
template <> const Precision Xomatcopy<float2>::precision_ = Precision::kComplexSingle;
template <> const Precision Xomatcopy<double2>::precision_ = Precision::kComplexDouble;

// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xomatcopy<T>::Xomatcopy(Queue &queue, EventPointer event, const std::string &name):
    Routine<T>(queue, event, name, {"Copy","Pad","Transpose","Padtranspose"}, precision_) {
  source_string_ =
    #include "../../kernels/level3/level3.opencl"
    #include "../../kernels/level3/copy_fast.opencl"
    #include "../../kernels/level3/copy_pad.opencl"
    #include "../../kernels/level3/transpose_fast.opencl"
    #include "../../kernels/level3/transpose_pad.opencl"
  ;
}

// =================================================================================================

// The main routine
template <typename T>
StatusCode Xomatcopy<T>::DoOmatcopy(const Layout layout, const Transpose a_transpose,
                                    const size_t m, const size_t n, const T alpha,
                                    const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                                    const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld) {

  // Makes sure all dimensions are larger than zero
  if ((m == 0) || (n == 0)) { return StatusCode::kInvalidDimension; }

  // Determines whether to transpose the matrix A
  const auto transpose = (a_transpose != Transpose::kNo);

  // In case of complex data-types, the transpose can also become a conjugate transpose
  const auto conjugate = (a_transpose == Transpose::kConjugate);

  // Computes the dimensions of the two matrices
  const auto rotated = (layout == Layout::kRowMajor);
  const auto a_one = (rotated) ? n : m;
  const auto a_two = (rotated) ? m : n;
  const auto b_one = (transpose) ? a_two : a_one;
  const auto b_two = (transpose) ? a_one : a_two;

  // Tests the matrices for validity, first from a perspective of the OpenCL buffers and their
  // sizes, and then from a perspective of parameter values (e.g. m, n). Tests whether the OpenCL
  // buffers are valid and non-zero and whether the OpenCL buffers have sufficient storage space.
  // Also tests that the leading dimensions of:
  //    matrix A cannot be less than N when rotated, or less than M when not-rotated
  //    matrix B cannot be less than M when rotated, or less than N when not-rotated
  auto status = TestMatrixA(a_one, a_two, a_buffer, a_offset, a_ld);
  if (ErrorIn(status)) { return status; }
  status = TestMatrixB(b_one, b_two, b_buffer, b_offset, b_ld);
  if (ErrorIn(status)) { return status; }

  // Loads the program from the database
  const auto program = GetProgramFromCache();

  auto emptyEventList = std::vector<Event>();
  status = PadCopyTransposeMatrix(event_, emptyEventList,
                                  a_one, a_two, a_ld, a_offset, a_buffer,
                                  b_one, b_two, b_ld, b_offset, b_buffer,
                                  alpha, program, false, transpose, conjugate);
  if (ErrorIn(status)) { return status; }

  return StatusCode::kSuccess;
}

// =================================================================================================

// Compiles the templated class
template class Xomatcopy<half>;
template class Xomatcopy<float>;
template class Xomatcopy<double>;
template class Xomatcopy<float2>;
template class Xomatcopy<double2>;

// =================================================================================================
} // namespace clblast
