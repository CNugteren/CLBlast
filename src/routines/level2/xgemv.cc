
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xgemv class (see the header for information about the class).
//
// =================================================================================================

#include "internal/routines/level2/xgemv.h"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Specific implementations to get the memory-type based on a template argument
template <> const Precision Xgemv<float>::precision_ = Precision::kSingle;
template <> const Precision Xgemv<double>::precision_ = Precision::kDouble;
template <> const Precision Xgemv<float2>::precision_ = Precision::kComplexSingle;
template <> const Precision Xgemv<double2>::precision_ = Precision::kComplexDouble;

// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xgemv<T>::Xgemv(CommandQueue &queue, Event &event):
    Routine(queue, event, {"Xgemv"}, precision_) {
}

// =================================================================================================

// The main routine
template <typename T>
StatusCode Xgemv<T>::DoGemv(const Layout layout, const Transpose a_transpose,
                            const size_t m, const size_t n,
                            const T alpha,
                            const Buffer &a_buffer, const size_t a_offset, const size_t a_ld,
                            const Buffer &x_buffer, const size_t x_offset, const size_t x_inc,
                            const T beta,
                            const Buffer &y_buffer, const size_t y_offset, const size_t y_inc) {

  // Makes sure all dimensions are larger than zero
  if (m == 0 || n == 0) { return StatusCode::kInvalidDimension; }

  // Computes whether or not the matrix has an alternative layout (row or column-major).
  auto a_altlayout = (layout == Layout::kRowMajor);
  auto a_one = (a_altlayout) ? n : m;
  auto a_two = (a_altlayout) ? m : n;

  // Swap m and n if the matrix is transposed
  auto a_transposed = (a_transpose != Transpose::kNo);
  auto m_real = (a_transposed) ? n : m;
  auto n_real = (a_transposed) ? m : n;

  // Determines whether the kernel needs to perform rotated access ('^' is the XOR operator)
  auto a_rotated = a_transposed ^ a_altlayout;

  // In case of complex data-types, the transpose can also become a conjugate transpose
  auto a_conjugate = (a_transpose == Transpose::kConjugate);

  // Tests the matrix and the vectors for validity
  auto status = TestMatrixA(a_one, a_two, a_buffer, a_offset, a_ld, sizeof(T));
  if (ErrorIn(status)) { return status; }
  status = TestVectorX(n_real, x_buffer, x_offset, x_inc, sizeof(T));
  if (ErrorIn(status)) { return status; }
  status = TestVectorY(m_real, y_buffer, y_offset, y_inc, sizeof(T));
  if (ErrorIn(status)) { return status; }

  // Determines whether or not the fast-version can be used
  bool use_fast_kernel = (a_offset == 0) && (a_rotated == 0) && (a_conjugate == 0) &&
                         IsMultiple(m, db_["WGS2"]*db_["WPT2"]) &&
                         IsMultiple(n, db_["WGS2"]) &&
                         IsMultiple(a_ld, db_["VW2"]);
  bool use_fast_kernel_rot = (a_offset == 0) && (a_rotated == 1) && (a_conjugate == 0) &&
                             IsMultiple(m, db_["WGS3"]*db_["WPT3"]) &&
                             IsMultiple(n, db_["WGS3"]) &&
                             IsMultiple(a_ld, db_["VW3"]);

  // If possible, run the fast-version (rotated or non-rotated) of the kernel
  auto kernel_name = "Xgemv";
  auto m_ceiled = Ceil(m_real, db_["WGS1"]*db_["WPT1"]);
  auto global_size = m_ceiled / db_["WPT1"];
  auto local_size = db_["WGS1"];
  if (use_fast_kernel) {
    kernel_name = "XgemvFast";
    global_size = m_real / db_["WPT2"];
    local_size = db_["WGS2"];
  }
  if (use_fast_kernel_rot) {
    kernel_name = "XgemvFastRot";
    global_size = m_real / db_["WPT3"];
    local_size = db_["WGS3"];
  }

  // Retrieves the Xgemv kernel from the compiled binary
  try {
    auto& program = GetProgramFromCache();
    auto kernel = Kernel(program, kernel_name);

    // Sets the kernel arguments
    kernel.SetArgument(0, static_cast<int>(m_real));
    kernel.SetArgument(1, static_cast<int>(n_real));
    kernel.SetArgument(2, alpha);
    kernel.SetArgument(3, beta);
    kernel.SetArgument(4, static_cast<int>(a_rotated));
    kernel.SetArgument(5, a_buffer());
    kernel.SetArgument(6, static_cast<int>(a_offset));
    kernel.SetArgument(7, static_cast<int>(a_ld));
    kernel.SetArgument(8, x_buffer());
    kernel.SetArgument(9, static_cast<int>(x_offset));
    kernel.SetArgument(10, static_cast<int>(x_inc));
    kernel.SetArgument(11, y_buffer());
    kernel.SetArgument(12, static_cast<int>(y_offset));
    kernel.SetArgument(13, static_cast<int>(y_inc));
    kernel.SetArgument(14, static_cast<int>(a_conjugate));

    // Launches the kernel
    auto global = std::vector<size_t>{global_size};
    auto local = std::vector<size_t>{local_size};
    status = RunKernel(kernel, global, local);
    if (ErrorIn(status)) { return status; }

    // Waits for all kernels to finish
    queue_.Finish();

    // Succesfully finished the computation
    return StatusCode::kSuccess;
  } catch (...) { return StatusCode::kInvalidKernel; }
}

// =================================================================================================

// Compiles the templated class
template class Xgemv<float>;
template class Xgemv<double>;
template class Xgemv<float2>;
template class Xgemv<double2>;

// =================================================================================================
} // namespace clblast
