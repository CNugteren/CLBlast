
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xher2k class (see the header for information about the class).
//
// =================================================================================================

#include "internal/routines/level3/xher2k.h"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Specific implementations to get the memory-type based on a template argument
template <> const Precision Xher2k<float2,float>::precision_ = Precision::kComplexSingle;
template <> const Precision Xher2k<double2,double>::precision_ = Precision::kComplexDouble;

// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T, typename U>
Xher2k<T,U>::Xher2k(Queue &queue, EventPointer event, const std::string &name):
    Routine<T>(queue, event, name, {"Copy","Pad","Transpose","Padtranspose","Xgemm"}, precision_) {
  source_string_ =
    #include "../../kernels/level3/level3.opencl"
    #include "../../kernels/level3/copy_fast.opencl"
    #include "../../kernels/level3/copy_pad.opencl"
    #include "../../kernels/level3/transpose_fast.opencl"
    #include "../../kernels/level3/transpose_pad.opencl"
    #include "../../kernels/level3/xgemm_part1.opencl"
    #include "../../kernels/level3/xgemm_part2.opencl"
  ;
}

// =================================================================================================

// The main routine
template <typename T, typename U>
StatusCode Xher2k<T,U>::DoHer2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                                const size_t n, const size_t k,
                                const T alpha,
                                const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                                const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld,
                                const U beta,
                                const Buffer<T> &c_buffer, const size_t c_offset, const size_t c_ld) {

  // Makes sure all dimensions are larger than zero
  if ((n == 0) || (k == 0) ) { return StatusCode::kInvalidDimension; }

  // Determines whether to apply the conjugate transpose to matrix B (argument: no transpose) or
  // to matrix A (argument: conjugate transpose)
  auto ab_conjugate = (ab_transpose != Transpose::kNo);

  // Computes whether or not the matrices are transposed in memory. This is based on their layout
  // (row or column-major) and whether or not they are requested to be pre-transposed.
  auto ab_rotated = (layout == Layout::kColMajor && ab_conjugate) ||
                    (layout == Layout::kRowMajor && !ab_conjugate);
  auto c_rotated = (layout == Layout::kRowMajor);

  // Computes the first and second dimensions of the A and B matrices taking the layout into account
  auto ab_one = (ab_rotated) ? k : n;
  auto ab_two = (ab_rotated) ? n : k;

  // Tests the matrices (A, B, C) for validity, first from a perspective of the OpenCL buffers and
  // their sizes, and then from a perspective of parameter values (e.g. n, k). Tests whether the
  // OpenCL buffers are valid and non-zero and whether the OpenCL buffers have sufficient storage
  // space. Also tests that the leading dimensions of:
  //    matrix A cannot be less than N when rotated, or less than K when not-rotated
  //    matrix B cannot be less than N when rotated, or less than K when not-rotated
  //    matrix C cannot be less than N
  auto status = TestMatrixA(ab_one, ab_two, a_buffer, a_offset, a_ld, sizeof(T));
  if (ErrorIn(status)) { return status; }
  status = TestMatrixB(ab_one, ab_two, b_buffer, b_offset, b_ld, sizeof(T));
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
    auto a1_no_temp = ab_one == n_ceiled && ab_two == k_ceiled && a_ld == n_ceiled && a_offset == 0 &&
                      ab_rotated == false && ab_conjugate == false;
    auto a2_no_temp = ab_one == n_ceiled && ab_two == k_ceiled && a_ld == n_ceiled && a_offset == 0 &&
                      ab_rotated == false && ab_conjugate == true;
    auto b1_no_temp = ab_one == n_ceiled && ab_two == k_ceiled && b_ld == n_ceiled && b_offset == 0 &&
                      ab_rotated == false && ab_conjugate == false;
    auto b2_no_temp = ab_one == n_ceiled && ab_two == k_ceiled && b_ld == n_ceiled && b_offset == 0 &&
                      ab_rotated == false && ab_conjugate == true;

    // Creates the temporary matrices
    auto a1_temp = (a1_no_temp) ? a_buffer : Buffer<T>(context_, k_ceiled*n_ceiled);
    auto a2_temp = (a2_no_temp) ? a_buffer : Buffer<T>(context_, k_ceiled*n_ceiled);
    auto b1_temp = (b1_no_temp) ? b_buffer : Buffer<T>(context_, k_ceiled*n_ceiled);
    auto b2_temp = (b2_no_temp) ? b_buffer : Buffer<T>(context_, k_ceiled*n_ceiled);
    auto c_temp = Buffer<T>(context_, n_ceiled*n_ceiled);

    // Upload the scalar arguments as constant buffers to the device (needed for half-precision)
    auto complex_beta = T{beta, static_cast<U>(0.0)};
    auto alpha_buffer = Buffer<T>(context_, 1);
    auto beta_buffer = Buffer<T>(context_, 1);
    alpha_buffer.Write(queue_, 1, &alpha);
    beta_buffer.Write(queue_, 1, &complex_beta);

    // Events of all kernels (including pre/post processing kernels)
    auto eventWaitList = std::vector<Event>();
    auto emptyEventList = std::vector<Event>();

    // Runs the pre-processing kernels. This transposes the matrices A and B, but also pads zeros to
    // to fill it up until it reaches a certain multiple of size (kernel parameter dependent). In
    // case nothing has to be done, these kernels can be skipped.
    if (!a1_no_temp) {
      auto eventProcessA1 = Event();
      status = PadCopyTransposeMatrix(eventProcessA1.pointer(), emptyEventList,
                                      ab_one, ab_two, a_ld, a_offset, a_buffer,
                                      n_ceiled, k_ceiled, n_ceiled, 0, a1_temp,
                                      program, true, ab_rotated, ab_conjugate);
      eventWaitList.push_back(eventProcessA1);
      if (ErrorIn(status)) { return status; }
    }
    if (!a2_no_temp) {
      auto eventProcessA2 = Event();
      status = PadCopyTransposeMatrix(eventProcessA2.pointer(), emptyEventList,
                                      ab_one, ab_two, a_ld, a_offset, a_buffer,
                                      n_ceiled, k_ceiled, n_ceiled, 0, a2_temp,
                                      program, true, ab_rotated, !ab_conjugate);
      eventWaitList.push_back(eventProcessA2);
      if (ErrorIn(status)) { return status; }
    }
    if (!b1_no_temp) {
      auto eventProcessB1 = Event();
      status = PadCopyTransposeMatrix(eventProcessB1.pointer(), emptyEventList,
                                      ab_one, ab_two, b_ld, b_offset, b_buffer,
                                      n_ceiled, k_ceiled, n_ceiled, 0, b1_temp,
                                      program, true, ab_rotated, ab_conjugate);
      eventWaitList.push_back(eventProcessB1);
      if (ErrorIn(status)) { return status; }
    }
    if (!b2_no_temp) {
      auto eventProcessB2 = Event();
      status = PadCopyTransposeMatrix(eventProcessB2.pointer(), emptyEventList,
                                      ab_one, ab_two, b_ld, b_offset, b_buffer,
                                      n_ceiled, k_ceiled, n_ceiled, 0, b2_temp,
                                      program, true, ab_rotated, !ab_conjugate);
      eventWaitList.push_back(eventProcessB2);
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
      kernel.SetArgument(0, static_cast<int>(n_ceiled));
      kernel.SetArgument(1, static_cast<int>(k_ceiled));
      kernel.SetArgument(2, alpha_buffer());
      kernel.SetArgument(3, beta_buffer());
      kernel.SetArgument(4, a1_temp());
      kernel.SetArgument(5, b2_temp());
      kernel.SetArgument(6, c_temp());

      // Computes the global and local thread sizes
      auto global = std::vector<size_t>{
        (n_ceiled * db_["MDIMC"]) / db_["MWG"],
        (n_ceiled * db_["NDIMC"]) / db_["NWG"]
      };
      auto local = std::vector<size_t>{db_["MDIMC"], db_["NDIMC"]};

      // Launches the kernel
      auto eventKernel1 = Event();
      status = RunKernel(kernel, global, local, eventKernel1.pointer(), eventWaitList);
      if (ErrorIn(status)) { return status; }
      eventWaitList.push_back(eventKernel1);

      // Swaps the arguments for matrices A and B, sets 'beta' to 1, and conjugate alpha
      auto conjugate_alpha = T{alpha.real(), -alpha.imag()};
      auto complex_one = T{static_cast<U>(1.0), static_cast<U>(0.0)};
      alpha_buffer.Write(queue_, 1, &conjugate_alpha);
      beta_buffer.Write(queue_, 1, &complex_one);
      kernel.SetArgument(2, alpha_buffer());
      kernel.SetArgument(3, beta_buffer());
      kernel.SetArgument(4, b1_temp());
      kernel.SetArgument(5, a2_temp());

      // Runs the kernel again
      auto eventKernel2 = Event();
      status = RunKernel(kernel, global, local, eventKernel2.pointer(), eventWaitList);
      if (ErrorIn(status)) { return status; }
      eventWaitList.push_back(eventKernel2);

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
template class Xher2k<float2,float>;
template class Xher2k<double2,double>;

// =================================================================================================
} // namespace clblast
