
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
    Routine(queue, event, name, {"XgemmDirect"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level3/xgemm_direct_part1.opencl"
    #include "../../kernels/level3/xgemm_direct_part2.opencl"
    #include "../../kernels/level3/xgemm_direct_part3.opencl"
    , // separated in multiple parts to prevent C1091 in MSVC 2013
    #include "../../kernels/level3/xgemm_direct_batched.opencl"
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

  // Upload the arguments to the device
  std::vector<int> a_offsets_int(a_offsets.begin(), a_offsets.end());
  std::vector<int> b_offsets_int(b_offsets.begin(), b_offsets.end());
  std::vector<int> c_offsets_int(c_offsets.begin(), c_offsets.end());
  auto a_offsets_device = Buffer<int>(context_, BufferAccess::kReadOnly, batch_count);
  auto b_offsets_device = Buffer<int>(context_, BufferAccess::kReadOnly, batch_count);
  auto c_offsets_device = Buffer<int>(context_, BufferAccess::kReadOnly, batch_count);
  auto alphas_device = Buffer<T>(context_, BufferAccess::kReadOnly, batch_count);
  auto betas_device = Buffer<T>(context_, BufferAccess::kReadOnly, batch_count);
  a_offsets_device.Write(queue_, batch_count, a_offsets_int);
  b_offsets_device.Write(queue_, batch_count, b_offsets_int);
  c_offsets_device.Write(queue_, batch_count, c_offsets_int);
  alphas_device.Write(queue_, batch_count, alphas);
  betas_device.Write(queue_, batch_count, betas);

  // Retrieves the proper XgemmDirect kernel from the compiled binary
  const auto name = (a_do_transpose) ? (b_do_transpose ? "XgemmDirectBatchedTT" : "XgemmDirectBatchedTN") :
                                       (b_do_transpose ? "XgemmDirectBatchedNT" : "XgemmDirectBatchedNN");
  auto kernel = Kernel(program_, name);

  // Sets the kernel arguments
  kernel.SetArgument(0, static_cast<int>(m));
  kernel.SetArgument(1, static_cast<int>(n));
  kernel.SetArgument(2, static_cast<int>(k));
  kernel.SetArgument(3, alphas_device());
  kernel.SetArgument(4, betas_device());
  kernel.SetArgument(5, a_buffer());
  kernel.SetArgument(6, a_offsets_device());
  kernel.SetArgument(7, static_cast<int>(a_ld));
  kernel.SetArgument(8, b_buffer());
  kernel.SetArgument(9, b_offsets_device());
  kernel.SetArgument(10, static_cast<int>(b_ld));
  kernel.SetArgument(11, c_buffer());
  kernel.SetArgument(12, c_offsets_device());
  kernel.SetArgument(13, static_cast<int>(c_ld));
  kernel.SetArgument(14, static_cast<int>(c_do_transpose));
  kernel.SetArgument(15, static_cast<int>(a_conjugate));
  kernel.SetArgument(16, static_cast<int>(b_conjugate));

  // Computes the global and local thread sizes
  const auto m_ceiled = Ceil(m, db_["WGD"]);
  const auto n_ceiled = Ceil(n, db_["WGD"]);
  const auto global = std::vector<size_t>{
    (m_ceiled * db_["MDIMCD"]) / db_["WGD"],
    (n_ceiled * db_["NDIMCD"]) / db_["WGD"],
    batch_count
  };
  const auto local = std::vector<size_t>{db_["MDIMCD"], db_["NDIMCD"], 1};

  // Launches the kernel
  RunKernel(kernel, queue_, device_, global, local, event_);
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
