
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

#include "routines/level2/xgemv.hpp"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xgemv<T>::Xgemv(Queue &queue, EventPointer event, const std::string &name):
    Routine(queue, event, name, {"Xgemv", "XgemvFast", "XgemvFastRot", "TrsvRoutine"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level2/xgemv.opencl"
    #include "../../kernels/level2/xgemv_fast.opencl"
    #include "../../kernels/level2/xtrsv.opencl"
    }) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xgemv<T>::DoGemv(const Layout layout, const Transpose a_transpose,
                      const size_t m, const size_t n,
                      const T alpha,
                      const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                      const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                      const T beta,
                      const Buffer<T> &y_buffer, const size_t y_offset, const size_t y_inc) {

  // Performs the matrix-vector multiplication
  MatVec(layout, a_transpose,
         m, n, alpha,
         a_buffer, a_offset, a_ld,
         x_buffer, x_offset, x_inc, beta,
         y_buffer, y_offset, y_inc,
         true, true,
         0, false, 0, 0); // N/A for this routine
}

// =================================================================================================

// The generic implementation, also suited for other (non general) matrix-vector multiplications
template <typename T>
void Xgemv<T>::MatVec(const Layout layout, const Transpose a_transpose,
                      const size_t m, const size_t n,
                      const T alpha,
                      const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                      const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                      const T beta,
                      const Buffer<T> &y_buffer, const size_t y_offset, const size_t y_inc,
                      bool fast_kernel, bool fast_kernel_rot,
                      const size_t parameter, const bool packed,
                      const size_t kl, const size_t ku) {

  // Makes sure all dimensions are larger than zero
  if (m == 0 || n == 0) { throw BLASError(StatusCode::kInvalidDimension); }

  // Computes whether or not the matrix has an alternative layout (row or column-major).
  const auto a_altlayout = (layout == Layout::kRowMajor);
  auto a_one = (a_altlayout) ? n : m;
  const auto a_two = (a_altlayout) ? m : n;

  // Swap m and n if the matrix is transposed
  const auto a_transposed = (a_transpose != Transpose::kNo);
  const auto m_real = (a_transposed) ? n : m;
  const auto n_real = (a_transposed) ? m : n;

  // Special adjustments for banded matrices
  if (kl != 0 || ku != 0) {
    a_one = kl+ku+1;
  }

  // Determines whether the kernel needs to perform rotated access ('^' is the XOR operator)
  const auto a_rotated = a_transposed ^ a_altlayout;

  // In case of complex data-types, the transpose can also become a conjugate transpose
  const auto a_conjugate = (a_transpose == Transpose::kConjugate);

  // Tests the matrix and the vectors for validity
  if (packed) { TestMatrixAP(n, a_buffer, a_offset); }
  else { TestMatrixA(a_one, a_two, a_buffer, a_offset, a_ld); }
  TestVectorX(n_real, x_buffer, x_offset, x_inc);
  TestVectorY(m_real, y_buffer, y_offset, y_inc);

  // Determines whether or not the fast-version can be used
  fast_kernel = fast_kernel && (a_offset == 0) && (a_rotated == 0) && (a_conjugate == 0) &&
                IsMultiple(m, db_["WGS2"]*db_["WPT2"]) &&
                IsMultiple(n, db_["WGS2"]) &&
                IsMultiple(a_ld, db_["VW2"]);
  fast_kernel_rot = fast_kernel_rot && (a_offset == 0) && (a_rotated == 1) && (a_conjugate == 0) &&
                    IsMultiple(m, db_["WGS3"]*db_["WPT3"]) &&
                    IsMultiple(n, db_["WGS3"]) &&
                    IsMultiple(a_ld, db_["VW3"]);

  // If possible, run the fast-version (rotated or non-rotated) of the kernel
  auto kernel_name = std::string{"Xgemv"};
  const auto m_ceiled = Ceil(m_real, db_["WGS1"]*db_["WPT1"]);
  auto global_size = m_ceiled / db_["WPT1"];
  auto local_size = db_["WGS1"];
  if (fast_kernel) {
    kernel_name = "XgemvFast";
    global_size = m_real / db_["WPT2"];
    local_size = db_["WGS2"];
  }
  if (fast_kernel_rot) {
    kernel_name = "XgemvFastRot";
    global_size = m_real;
    local_size = db_["WGS3"];
  }

  // Retrieves the Xgemv kernel from the compiled binary
  auto kernel = Kernel(program_, kernel_name);

  // Sets the kernel arguments
  kernel.SetArgument(0, static_cast<int>(m_real));
  kernel.SetArgument(1, static_cast<int>(n_real));
  kernel.SetArgument(2, GetRealArg(alpha));
  kernel.SetArgument(3, GetRealArg(beta));
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
  kernel.SetArgument(15, static_cast<int>(parameter)); // extra parameter used for symm/herm
  kernel.SetArgument(16, static_cast<int>(kl)); // only used for banded matrices
  kernel.SetArgument(17, static_cast<int>(ku)); // only used for banded matrices

  // Launches the kernel
  auto global = std::vector<size_t>{global_size};
  auto local = std::vector<size_t>{local_size};
  RunKernel(kernel, queue_, device_, global, local, event_);
}

// =================================================================================================

// Compiles the templated class
template class Xgemv<half>;
template class Xgemv<float>;
template class Xgemv<double>;
template class Xgemv<float2>;
template class Xgemv<double2>;

// =================================================================================================
} // namespace clblast
