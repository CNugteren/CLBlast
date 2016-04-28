
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xherk class (see the header for information about the class).
//
// =================================================================================================

#include "internal/routines/level3/xherk.h"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Specific implementations to get the memory-type based on a template argument
template <> const Precision Xherk<float2,float>::precision_ = Precision::kComplexSingle;
template <> const Precision Xherk<double2,double>::precision_ = Precision::kComplexDouble;

// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T, typename U>
Xherk<T,U>::Xherk(Queue &queue, EventPointer event, const std::string &name):
    Routine<T>(queue, event, name, {"Copy","Pad","Transpose","Padtranspose","Xgemm"}, precision_) {
  source_string_ =
    #include "../../kernels/level3/copy.opencl"
    #include "../../kernels/level3/pad.opencl"
    #include "../../kernels/level3/transpose.opencl"
    #include "../../kernels/level3/padtranspose.opencl"
    #include "../../kernels/level3/xgemm_part1.opencl"
    #include "../../kernels/level3/xgemm_part2.opencl"
  ;
}

// =================================================================================================

// The main routine
template <typename T, typename U>
StatusCode Xherk<T,U>::DoHerk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                              const size_t n, const size_t k,
                              const U alpha,
                              const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                              const U beta,
                              const Buffer<T> &c_buffer, const size_t c_offset, const size_t c_ld) {

  // Makes sure all dimensions are larger than zero
  if ((n == 0) || (k == 0) ) { return StatusCode::kInvalidDimension; }

  // Determines whether to apply the conjugate transpose to matrix B (argument: no transpose) or
  // to matrix A (argument: conjugate transpose)
  auto a_conjugate = (a_transpose != Transpose::kNo);
  auto b_conjugate = (a_transpose == Transpose::kNo);

  // Computes whether or not the matrices are transposed in memory. This is based on their layout
  // (row or column-major) and whether or not they are requested to be pre-transposed.
  auto a_rotated = (layout == Layout::kColMajor && a_conjugate) ||
                   (layout == Layout::kRowMajor && !a_conjugate);
  auto c_rotated = (layout == Layout::kRowMajor);

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

  // The padded/transposed input/output matrices: if memory allocation fails, throw an exception
  try {

    // Loads the program from the database
    const auto program = GetProgramFromCache();

    // Determines whether or not temporary matrices are needed
    auto a_no_temp = a_one == n_ceiled && a_two == k_ceiled && a_ld == n_ceiled && a_offset == 0 &&
                     a_rotated == false && a_conjugate == false;
    auto b_no_temp = a_one == n_ceiled && a_two == k_ceiled && a_ld == n_ceiled && a_offset == 0 &&
                     a_rotated == false && b_conjugate == false;

    // Creates the temporary matrices
    auto a_temp = (a_no_temp) ? a_buffer : Buffer<T>(context_, k_ceiled*n_ceiled);
    auto b_temp = (b_no_temp) ? a_buffer : Buffer<T>(context_, k_ceiled*n_ceiled);
    auto c_temp = Buffer<T>(context_, n_ceiled*n_ceiled);

    // Events of all kernels (including pre/post processing kernels)
    auto eventWaitList = std::vector<Event>();
    auto emptyEventList = std::vector<Event>();

    // Runs the pre-processing kernel for matrix A. This transposes the matrix, but also pads zeros
    // to fill it up until it reaches a certain multiple of size (kernel parameter dependent). In
    // case nothing has to be done, these kernels can be skipped. Two copies are created.
    if (!a_no_temp) {
      auto eventProcessA = Event();
      status = PadCopyTransposeMatrix(eventProcessA.pointer(), emptyEventList,
                                      a_one, a_two, a_ld, a_offset, a_buffer,
                                      n_ceiled, k_ceiled, n_ceiled, 0, a_temp,
                                      program, true, a_rotated, a_conjugate);
      eventWaitList.push_back(eventProcessA);
      if (ErrorIn(status)) { return status; }
    }
    if (!b_no_temp) {
      auto eventProcessB = Event();
      status = PadCopyTransposeMatrix(eventProcessB.pointer(), emptyEventList,
                                      a_one, a_two, a_ld, a_offset, a_buffer,
                                      n_ceiled, k_ceiled, n_ceiled, 0, b_temp,
                                      program, true, a_rotated, b_conjugate);
      eventWaitList.push_back(eventProcessB);
      if (ErrorIn(status)) { return status; }
    }

    // Furthermore, also creates a (possibly padded) copy of matrix C, since it is not allowed to
    // modify the other triangle.
    auto eventProcessC = Event();
    status = PadCopyTransposeMatrix(eventProcessC.pointer(), emptyEventList,
                                    n, n, c_ld, c_offset, c_buffer,
                                    n_ceiled, n_ceiled, n_ceiled, 0, c_temp,
                                    program, true, c_rotated, false);
    eventWaitList.push_back(eventProcessC);
    if (ErrorIn(status)) { return status; }

    // Retrieves the XgemmUpper or XgemmLower kernel from the compiled binary
    try {
      auto kernel = Kernel(program, kernel_name);

      // Sets the kernel arguments
      auto complex_alpha = T{alpha, static_cast<U>(0.0)};
      auto complex_beta = T{beta, static_cast<U>(0.0)};
      kernel.SetArgument(0, static_cast<int>(n_ceiled));
      kernel.SetArgument(1, static_cast<int>(k_ceiled));
      kernel.SetArgument(2, complex_alpha);
      kernel.SetArgument(3, complex_beta);
      kernel.SetArgument(4, a_temp());
      kernel.SetArgument(5, b_temp());
      kernel.SetArgument(6, c_temp());

      // Computes the global and local thread sizes
      auto global = std::vector<size_t>{
        (n_ceiled * db_["MDIMC"]) / db_["MWG"],
        (n_ceiled * db_["NDIMC"]) / db_["NWG"]
      };
      auto local = std::vector<size_t>{db_["MDIMC"], db_["NDIMC"]};

      // Launches the kernel
      auto eventKernel = Event();
      status = RunKernel(kernel, global, local, eventKernel.pointer(), eventWaitList);
      if (ErrorIn(status)) { return status; }
      eventWaitList.push_back(eventKernel);

      // Runs the post-processing kernel
      auto upper = (triangle == Triangle::kUpper);
      auto lower = (triangle == Triangle::kLower);
      status = PadCopyTransposeMatrix(event_, eventWaitList,
                                      n_ceiled, n_ceiled, n_ceiled, 0, c_temp,
                                      n, n, c_ld, c_offset, c_buffer,
                                      program, false, c_rotated, false, upper, lower, true);
      if (ErrorIn(status)) { return status; }

      // Successfully finished the computation
      return StatusCode::kSuccess;
    } catch (...) { return StatusCode::kInvalidKernel; }
  } catch (...) { return StatusCode::kTempBufferAllocFailure; }
}

// =================================================================================================

// Compiles the templated class
template class Xherk<float2,float>;
template class Xherk<double2,double>;

// =================================================================================================
} // namespace clblast
