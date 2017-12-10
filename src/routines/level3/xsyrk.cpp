
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

#include "routines/level3/xsyrk.hpp"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xsyrk<T>::Xsyrk(Queue &queue, EventPointer event, const std::string &name):
    Routine(queue, event, name, {"Copy","Pad","Transpose","Padtranspose","Xgemm"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level3/level3.opencl"
    #include "../../kernels/level3/copy_fast.opencl"
    #include "../../kernels/level3/copy_pad.opencl"
    #include "../../kernels/level3/transpose_fast.opencl"
    #include "../../kernels/level3/transpose_pad.opencl"
    #include "../../kernels/level3/xgemm_part1.opencl"
    #include "../../kernels/level3/xgemm_part2.opencl"
    #include "../../kernels/level3/xgemm_part3.opencl"
    #include "../../kernels/level3/xgemm_part4.opencl"
    }) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xsyrk<T>::DoSyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                            const size_t n, const size_t k,
                            const T alpha,
                            const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                            const T beta,
                            const Buffer<T> &c_buffer, const size_t c_offset, const size_t c_ld) {

  // Makes sure all dimensions are larger than zero
  if ((n == 0) || (k == 0) ) { throw BLASError(StatusCode::kInvalidDimension); }

  // Computes whether or not the matrices are transposed in memory. This is based on their layout
  // (row or column-major) and whether or not they are requested to be pre-transposed.
  auto a_rotated = (layout == Layout::kColMajor && a_transpose != Transpose::kNo) ||
                   (layout == Layout::kRowMajor && a_transpose == Transpose::kNo);
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
  TestMatrixA(a_one, a_two, a_buffer, a_offset, a_ld);
  TestMatrixC(n, n, c_buffer, c_offset, c_ld);

  // Calculates the ceiled versions of n and k
  auto n_ceiled = Ceil(Ceil(n, db_["MWG"]), db_["NWG"]);
  auto k_ceiled = Ceil(k, db_["KWG"]);

  // Decides which kernel to run: the upper-triangular or lower-triangular version
  auto kernel_name = (triangle == Triangle::kUpper) ? "XgemmUpper" : "XgemmLower";

  // Determines whether or not temporary matrices are needed
  auto a_no_temp = a_one == n_ceiled && a_two == k_ceiled && a_ld == n_ceiled && a_offset == 0 &&
                   a_rotated == false;

  // Creates the temporary matrices
  auto a_temp = (a_no_temp) ? a_buffer : Buffer<T>(context_, k_ceiled*n_ceiled);
  auto c_temp = Buffer<T>(context_, n_ceiled*n_ceiled);

  // Events of all kernels (including pre/post processing kernels)
  auto eventWaitList = std::vector<Event>();
  auto emptyEventList = std::vector<Event>();

  // Runs the pre-processing kernel for matrix A. This transposes the matrix, but also pads zeros
  // to fill it up until it reaches a certain multiple of size (kernel parameter dependent). In
  // case nothing has to be done, these kernels can be skipped.
  if (!a_no_temp) {
    auto eventProcessA = Event();
    PadCopyTransposeMatrix(queue_, device_, db_, eventProcessA.pointer(), emptyEventList,
                           a_one, a_two, a_ld, a_offset, a_buffer,
                           n_ceiled, k_ceiled, n_ceiled, 0, a_temp,
                           ConstantOne<T>(), program_,
                           true, a_rotated, false);
    eventWaitList.push_back(eventProcessA);
  }

  // Furthermore, also creates a (possibly padded) copy of matrix C, since it is not allowed to
  // modify the other triangle.
  auto eventProcessC = Event();
  PadCopyTransposeMatrix(queue_, device_, db_, eventProcessC.pointer(), emptyEventList,
                         n, n, c_ld, c_offset, c_buffer,
                         n_ceiled, n_ceiled, n_ceiled, 0, c_temp,
                         ConstantOne<T>(), program_,
                         true, c_rotated, false);
  eventWaitList.push_back(eventProcessC);

  // Retrieves the XgemmUpper or XgemmLower kernel from the compiled binary
  auto kernel = Kernel(program_, kernel_name);

  // Sets the kernel arguments
  kernel.SetArgument(0, static_cast<int>(n_ceiled));
  kernel.SetArgument(1, static_cast<int>(k_ceiled));
  kernel.SetArgument(2, GetRealArg(alpha));
  kernel.SetArgument(3, GetRealArg(beta));
  kernel.SetArgument(4, a_temp());
  kernel.SetArgument(5, a_temp());
  kernel.SetArgument(6, c_temp());

  // Computes the global and local thread sizes
  auto global = std::vector<size_t>{
    (n_ceiled * db_["MDIMC"]) / db_["MWG"],
    (n_ceiled * db_["NDIMC"]) / db_["NWG"]
  };
  auto local = std::vector<size_t>{db_["MDIMC"], db_["NDIMC"]};

  // Launches the kernel
  auto eventKernel = Event();
  RunKernel(kernel, queue_, device_, global, local, eventKernel.pointer(), eventWaitList);
  eventWaitList.push_back(eventKernel);

  // Runs the post-processing kernel
  auto upper = (triangle == Triangle::kUpper);
  auto lower = (triangle == Triangle::kLower);
  PadCopyTransposeMatrix(queue_, device_, db_, event_, eventWaitList,
                         n_ceiled, n_ceiled, n_ceiled, 0, c_temp,
                         n, n, c_ld, c_offset, c_buffer,
                         ConstantOne<T>(), program_,
                         false, c_rotated, false, upper, lower, false);
}

// =================================================================================================

// Compiles the templated class
template class Xsyrk<half>;
template class Xsyrk<float>;
template class Xsyrk<double>;
template class Xsyrk<float2>;
template class Xsyrk<double2>;

// =================================================================================================
} // namespace clblast
