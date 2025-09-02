
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xgemm class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level3/xgemm.hpp"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xgemm<T>::Xgemm(Queue& queue, EventPointer event, const std::string& name)
    : Routine(queue, event, name, {"Copy", "Pad", "Transpose", "Padtranspose", "Xgemm", "XgemmDirect", "GemmRoutine"},
              PrecisionValue<T>(), {},
              {
#include "../../kernels/level3/level3.opencl"
// (comment to prevent auto-re-ordering)
#include "../../kernels/level3/convert_hermitian.opencl"
#include "../../kernels/level3/convert_symmetric.opencl"
#include "../../kernels/level3/convert_triangular.opencl"
#include "../../kernels/level3/copy_fast.opencl"
#include "../../kernels/level3/copy_pad.opencl"
#include "../../kernels/level3/transpose_fast.opencl"
#include "../../kernels/level3/transpose_pad.opencl"
                  ,  // separated in multiple parts to prevent C1091 in MSVC 2013
#include "../../kernels/level3/xgemm_direct_part1.opencl"
#include "../../kernels/level3/xgemm_direct_part2.opencl"
#include "../../kernels/level3/xgemm_direct_part3.opencl"
                  ,  // separated in multiple parts to prevent C1091 in MSVC 2013
#include "../../kernels/level3/xgemm_part1.opencl"
#include "../../kernels/level3/xgemm_part2.opencl"
                  ,  // separated in multiple parts to prevent C1091 in MSVC 2013
#include "../../kernels/level3/xgemm_part3.opencl"
#include "../../kernels/level3/xgemm_part4.opencl"
              }) {
}

// =================================================================================================

template <typename T>
size_t Xgemm<T>::GetTempSize(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                             const size_t m, const size_t n, const size_t k, const size_t a_offset, const size_t a_ld,
                             const size_t b_offset, const size_t b_ld, const size_t c_offset, const size_t c_ld,
                             const size_t mwg, const size_t nwg, const size_t kwg, const size_t gemm_kernel_id) {
  // Computes the transpose/conjugate options and sets the a/b/c sizes based on that
  bool a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate;
  size_t a_one, a_two, b_one, b_two, c_one, c_two;
  ProcessArguments(layout, a_transpose, b_transpose, m, n, k, a_one, a_two, b_one, b_two, c_one, c_two, a_do_transpose,
                   b_do_transpose, c_do_transpose, a_conjugate, b_conjugate, gemm_kernel_id);

  // Computes the first and second "internal" (ceiled) dimensions of the 3 matrices taking into account
  // whether the matrices need to be rotated or not for the kernel.
  size_t a_one_i, a_two_i, b_one_i, b_two_i, c_one_i, c_two_i;
  CalculateInternalDimensions(m, n, k, mwg, nwg, kwg, a_one_i, a_two_i, b_one_i, b_two_i, c_one_i, c_two_i,
                              gemm_kernel_id);

  // Determines whether or not temporary matrices are needed
  auto a_no_temp = NoTempBuffer(a_one, a_one_i, a_two, a_two_i, a_ld, a_offset, a_do_transpose, a_conjugate);
  auto b_no_temp = NoTempBuffer(b_one, b_one_i, b_two, b_two_i, b_ld, b_offset, b_do_transpose, b_conjugate);
  auto c_no_temp = NoTempBuffer(c_one, c_one_i, c_two, c_two_i, c_ld, c_offset, c_do_transpose, false);

  // Computes the sizes and offsets for (optional) temporary buffers for the 3 matrices
  auto b_temp_offset = size_t{0};
  auto c_temp_offset = size_t{0};
  return ComputeTempSize(a_no_temp, b_no_temp, c_no_temp, a_one_i * a_two_i, b_one_i * b_two_i, c_one_i * c_two_i,
                         b_temp_offset, c_temp_offset);
}

template <typename T>
bool Xgemm<T>::UseDirectKernel(const size_t m, const size_t n, const size_t k, const size_t min_indirect_size) {
  const auto m_n_k =
      static_cast<unsigned long long>(m) * static_cast<unsigned long long>(n) * static_cast<unsigned long long>(k);
  const auto min_indirect_size_ll = static_cast<unsigned long long>(min_indirect_size);
  const auto min_indirect_size_e3 = min_indirect_size_ll * min_indirect_size_ll * min_indirect_size_ll;
  return (m_n_k < min_indirect_size_e3);
}

template <typename T>
void Xgemm<T>::ProcessArguments(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                                const size_t m, const size_t n, const size_t k, size_t& a_one, size_t& a_two,
                                size_t& b_one, size_t& b_two, size_t& c_one, size_t& c_two, bool& a_do_transpose,
                                bool& b_do_transpose, bool& c_do_transpose, bool& a_conjugate, bool& b_conjugate,
                                const size_t gemm_kernel_id) {
  // Makes sure all dimensions are larger than zero
  if ((m == 0) || (n == 0) || (k == 0)) {
    throw BLASError(StatusCode::kInvalidDimension);
  }

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
  a_do_transpose = a_rotated != a_want_rotated_(gemm_kernel_id);
  b_do_transpose = b_rotated != b_want_rotated_(gemm_kernel_id);
  c_do_transpose = c_rotated != c_want_rotated_(gemm_kernel_id);

  // In case of complex data-types, the transpose can also become a conjugate transpose
  a_conjugate = (a_transpose == Transpose::kConjugate);
  b_conjugate = (b_transpose == Transpose::kConjugate);

  // Computes the first and second dimensions of the 3 matrices taking into account whether the
  // matrices are rotated or not
  a_one = (a_rotated) ? k : m;
  a_two = (a_rotated) ? m : k;
  b_one = (b_rotated) ? n : k;
  b_two = (b_rotated) ? k : n;
  c_one = (c_rotated) ? n : m;
  c_two = (c_rotated) ? m : n;
}

template <typename T>
size_t Xgemm<T>::ComputeTempSize(const bool a_no_temp, const bool b_no_temp, const bool c_no_temp, const size_t a_size,
                                 const size_t b_size, const size_t c_size, size_t& b_temp_offset,
                                 size_t& c_temp_offset) {
  auto temp_size = size_t{0};
  if (!a_no_temp) {
    temp_size += a_size;
  }
  if (!b_no_temp) {
    b_temp_offset = temp_size;
    temp_size += b_size;
  }
  if (!c_no_temp) {
    c_temp_offset = temp_size;
    temp_size += c_size;
  }
  return temp_size;
}

template <typename T>
void Xgemm<T>::CalculateInternalDimensions(const size_t m, const size_t n, const size_t k, const size_t mwg,
                                           const size_t nwg, const size_t kwg, size_t& a_one_i, size_t& a_two_i,
                                           size_t& b_one_i, size_t& b_two_i, size_t& c_one_i, size_t& c_two_i,
                                           const size_t gemm_kernel_id) {
  const auto global_divider_one = c_want_rotated_(gemm_kernel_id) ? nwg : mwg;
  const auto global_divider_two = c_want_rotated_(gemm_kernel_id) ? mwg : nwg;
  const auto m_ceiled = Ceil(m, global_divider_one);
  const auto n_ceiled = Ceil(n, global_divider_two);
  const auto k_ceiled = Ceil(k, kwg);
  a_one_i = (a_want_rotated_(gemm_kernel_id)) ? k_ceiled : m_ceiled;
  a_two_i = (a_want_rotated_(gemm_kernel_id)) ? m_ceiled : k_ceiled;
  b_one_i = (b_want_rotated_(gemm_kernel_id)) ? n_ceiled : k_ceiled;
  b_two_i = (b_want_rotated_(gemm_kernel_id)) ? k_ceiled : n_ceiled;
  c_one_i = (c_want_rotated_(gemm_kernel_id)) ? n_ceiled : m_ceiled;
  c_two_i = (c_want_rotated_(gemm_kernel_id)) ? m_ceiled : n_ceiled;
}

// The main routine
template <typename T>
void Xgemm<T>::DoGemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose, const size_t m,
                      const size_t n, const size_t k, const T alpha, const Buffer<T>& a_buffer, const size_t a_offset,
                      const size_t a_ld, const Buffer<T>& b_buffer, const size_t b_offset, const size_t b_ld,
                      const T beta, const Buffer<T>& c_buffer, const size_t c_offset, const size_t c_ld,
                      const Buffer<T>& temp_buffer, const bool temp_buffer_provided) {  // optional arguments

  // Two methods to choose from, select which one to run
  const auto do_gemm_direct = UseDirectKernel(m, n, k, db_["XGEMM_MIN_INDIRECT_SIZE"]);
  const auto gemm_kernel_id = (do_gemm_direct) ? 0 : db_["GEMMK"];

  // Computes the transpose/conjugate options and sets the a/b/c sizes based on that
  bool a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate;
  size_t a_one, a_two, b_one, b_two, c_one, c_two;
  ProcessArguments(layout, a_transpose, b_transpose, m, n, k, a_one, a_two, b_one, b_two, c_one, c_two, a_do_transpose,
                   b_do_transpose, c_do_transpose, a_conjugate, b_conjugate, gemm_kernel_id);

  // Tests three matrices (A, B, C) for validity, first from a perspective of the OpenCL buffers and
  // their sizes, and then from a perspective of parameter values (e.g. m, n, k). Tests whether the
  // OpenCL buffers are valid and non-zero and whether the OpenCL buffers have sufficient storage
  // space. Also tests that the leading dimensions of:
  //    matrix A cannot be less than K when rotated, or less than M when not-rotated
  //    matrix B cannot be less than N when rotated, or less than K when not-rotated
  //    matrix C cannot be less than N when rotated, or less than M when not-rotated
  TestMatrixA(a_one, a_two, a_buffer, a_offset, a_ld);
  TestMatrixB(b_one, b_two, b_buffer, b_offset, b_ld);
  TestMatrixC(c_one, c_two, c_buffer, c_offset, c_ld);

  // Selects which version of GEMM to run
  if (do_gemm_direct) {  // for small sizes (single kernel)
    GemmDirect(m, n, k, alpha, a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, beta, c_buffer, c_offset, c_ld,
               a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate);
  } else {  // for larger sizes (pre/post-processing plus a very fast kernel)
    GemmIndirect(m, n, k, alpha, a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, beta, c_buffer, c_offset, c_ld,
                 a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate, a_one, a_two, b_one, b_two,
                 c_one, c_two, temp_buffer, temp_buffer_provided);
  }
}

// =================================================================================================

// The indirect version of GEMM. This uses the faster but non-general kernel. It has specific
// requirements, but several pre and post-processing kernels take care of those. However, the
// overhead of these extra kernels might not be ideal for certain devices/arguments.
template <typename T>
void Xgemm<T>::GemmIndirect(const size_t m, const size_t n, const size_t k, const T alpha, const Buffer<T>& a_buffer,
                            const size_t a_offset, const size_t a_ld, const Buffer<T>& b_buffer, const size_t b_offset,
                            const size_t b_ld, const T beta, const Buffer<T>& c_buffer, const size_t c_offset,
                            const size_t c_ld, const bool a_do_transpose, const bool b_do_transpose,
                            const bool c_do_transpose, const bool a_conjugate, const bool b_conjugate,
                            const size_t a_one, const size_t a_two, const size_t b_one, const size_t b_two,
                            const size_t c_one, const size_t c_two, const Buffer<T>& temp_buffer,
                            const bool temp_buffer_provided) {
  // Calculates the ceiled versions of m, n, and k
  const auto global_divider_one = c_want_rotated_(db_["GEMMK"]) ? db_["NWG"] : db_["MWG"];
  const auto global_divider_two = c_want_rotated_(db_["GEMMK"]) ? db_["MWG"] : db_["NWG"];
  const auto m_ceiled = Ceil(m, global_divider_one);
  const auto n_ceiled = Ceil(n, global_divider_two);
  const auto k_ceiled = Ceil(k, db_["KWG"] * db_["KREG"]);

  // Computes the first and second "internal" (ceiled) dimensions of the 3 matrices taking into account
  // whether the matrices need to be rotated or not for the kernel.
  size_t a_one_i, a_two_i, b_one_i, b_two_i, c_one_i, c_two_i;
  CalculateInternalDimensions(m, n, k, db_["MWG"], db_["NWG"], db_["KWG"] * db_["KREG"], a_one_i, a_two_i, b_one_i,
                              b_two_i, c_one_i, c_two_i, db_["GEMMK"]);

  // Determines whether or not temporary matrices are needed
  auto a_no_temp = NoTempBuffer(a_one, a_one_i, a_two, a_two_i, a_ld, a_offset, a_do_transpose, a_conjugate);
  auto b_no_temp = NoTempBuffer(b_one, b_one_i, b_two, b_two_i, b_ld, b_offset, b_do_transpose, b_conjugate);
  auto c_no_temp = NoTempBuffer(c_one, c_one_i, c_two, c_two_i, c_ld, c_offset, c_do_transpose, false);

  // Computes the sizes and offsets for (optional) temporary buffers for the 3 matrices
  auto b_temp_offset = size_t{0};
  auto c_temp_offset = size_t{0};
  const auto temp_size = ComputeTempSize(a_no_temp, b_no_temp, c_no_temp, a_one_i * a_two_i, b_one_i * b_two_i,
                                         c_one_i * c_two_i, b_temp_offset, c_temp_offset);
  if (!IsMultiple(b_temp_offset, db_["VWN"])) {
    throw BLASError(StatusCode::kUnexpectedError);
  }
  if (!IsMultiple(c_temp_offset, db_["VWM"])) {
    throw BLASError(StatusCode::kUnexpectedError);
  }

  // Creates the buffer for the (optional) temporary matrices. Note that we use 'a_buffer' in case
  // when no temporary buffer is needed, but that's just to make it compile: it is never used.
  const auto temp_buffer_all =
      (temp_buffer_provided) ? temp_buffer : ((temp_size > 0) ? Buffer<T>(context_, temp_size) : a_buffer);

  // Verifies if the provided temporary buffer is large enough
  if (temp_buffer_provided) {
    const auto required_size = temp_size * sizeof(T);
    if (temp_buffer_all.GetSize() < required_size) {
      throw BLASError(StatusCode::kInsufficientMemoryTemp);
    }
  }

  // Sets the buffer pointers for (temp) matrices A, B, and C
  const auto a_temp = (a_no_temp) ? a_buffer : temp_buffer_all;
  const auto b_temp = (b_no_temp) ? b_buffer : temp_buffer_all;
  const auto c_temp = (c_no_temp) ? c_buffer : temp_buffer_all;

  // Events of all kernels (including pre/post processing kernels)
  auto eventWaitList = std::vector<Event>();
  auto emptyEventList = std::vector<Event>();

  // Runs the pre-processing kernel for matrix A. This transposes the matrix, but also pads zeros
  // to fill it up until it reaches a certain multiple of size (kernel parameter dependent). In
  // case nothing has to be done, these kernels can be skipped.
  if (!a_no_temp) {
    auto eventProcessA = Event();
    PadCopyTransposeMatrix(queue_, device_, db_, eventProcessA.pointer(), emptyEventList, a_one, a_two, a_ld, a_offset,
                           a_buffer, a_one_i, a_two_i, a_one_i, 0, a_temp, ConstantOne<T>(), program_, true,
                           a_do_transpose, a_conjugate);
    eventWaitList.push_back(eventProcessA);
  }

  // As above, but now for matrix B
  if (!b_no_temp) {
    auto eventProcessB = Event();
    PadCopyTransposeMatrix(queue_, device_, db_, eventProcessB.pointer(), emptyEventList, b_one, b_two, b_ld, b_offset,
                           b_buffer, b_one_i, b_two_i, b_one_i, b_temp_offset, b_temp, ConstantOne<T>(), program_, true,
                           b_do_transpose, b_conjugate);
    eventWaitList.push_back(eventProcessB);
  }

  // As above, but now for matrix C. This is only necessary if C is used both as input and output.
  if (!c_no_temp && beta != static_cast<T>(0)) {
    auto eventProcessC = Event();
    PadCopyTransposeMatrix(queue_, device_, db_, eventProcessC.pointer(), emptyEventList, c_one, c_two, c_ld, c_offset,
                           c_buffer, c_one_i, c_two_i, c_one_i, c_temp_offset, c_temp, ConstantOne<T>(), program_, true,
                           c_do_transpose, false);
    eventWaitList.push_back(eventProcessC);
  }

  // Retrieves the Xgemm kernel from the compiled binary
  auto kernel = Kernel(program_, "Xgemm");

  // Sets the kernel arguments
  kernel.SetArgument(0, static_cast<int>(m_ceiled));
  kernel.SetArgument(1, static_cast<int>(n_ceiled));
  kernel.SetArgument(2, static_cast<int>(k_ceiled));
  kernel.SetArgument(3, GetRealArg(alpha));
  kernel.SetArgument(4, GetRealArg(beta));
  kernel.SetArgument(5, a_temp());
  kernel.SetArgument(6, b_temp());
  kernel.SetArgument(7, c_temp());
  kernel.SetArgument(8, static_cast<int>(b_temp_offset / db_["VWN"]));
  kernel.SetArgument(9, static_cast<int>(c_temp_offset / db_["VWM"]));

  // Computes the global and local thread sizes
  const auto global = std::vector<size_t>{(c_one_i * db_["MDIMC"]) / db_["MWG"], (c_two_i * db_["NDIMC"]) / db_["NWG"]};
  const auto local = std::vector<size_t>{db_["MDIMC"], db_["NDIMC"]};

  // Launches the kernel
  auto eventKernel = Event();
  auto eventPointer = (!c_no_temp) ? eventKernel.pointer() : event_;
  RunKernel(kernel, queue_, device_, global, local, eventPointer, eventWaitList);

  // Runs the post-processing kernel if needed
  if (!c_no_temp) {
    eventWaitList.push_back(eventKernel);
    PadCopyTransposeMatrix(queue_, device_, db_, event_, eventWaitList, c_one_i, c_two_i, c_one_i, c_temp_offset,
                           c_temp, c_one, c_two, c_ld, c_offset, c_buffer, ConstantOne<T>(), program_, false,
                           c_do_transpose, false);
  }
}

// =================================================================================================

// The direct version of GEMM, requiring just one kernel, no pre or post-processing kernels.
template <typename T>
void Xgemm<T>::GemmDirect(const size_t m, const size_t n, const size_t k, const T alpha, const Buffer<T>& a_buffer,
                          const size_t a_offset, const size_t a_ld, const Buffer<T>& b_buffer, const size_t b_offset,
                          const size_t b_ld, const T beta, const Buffer<T>& c_buffer, const size_t c_offset,
                          const size_t c_ld, const bool a_do_transpose, const bool b_do_transpose,
                          const bool c_do_transpose, const bool a_conjugate, const bool b_conjugate) {
  // Retrieves the proper XgemmDirect kernel from the compiled binary
  const auto name = (a_do_transpose) ? (b_do_transpose ? "XgemmDirectTT" : "XgemmDirectTN")
                                     : (b_do_transpose ? "XgemmDirectNT" : "XgemmDirectNN");
  auto kernel = Kernel(program_, name);

  // Sets the kernel arguments
  kernel.SetArgument(0, static_cast<int>(m));
  kernel.SetArgument(1, static_cast<int>(n));
  kernel.SetArgument(2, static_cast<int>(k));
  kernel.SetArgument(3, GetRealArg(alpha));
  kernel.SetArgument(4, GetRealArg(beta));
  kernel.SetArgument(5, a_buffer());
  kernel.SetArgument(6, static_cast<int>(a_offset));
  kernel.SetArgument(7, static_cast<int>(a_ld));
  kernel.SetArgument(8, b_buffer());
  kernel.SetArgument(9, static_cast<int>(b_offset));
  kernel.SetArgument(10, static_cast<int>(b_ld));
  kernel.SetArgument(11, c_buffer());
  kernel.SetArgument(12, static_cast<int>(c_offset));
  kernel.SetArgument(13, static_cast<int>(c_ld));
  kernel.SetArgument(14, static_cast<int>(c_do_transpose));
  kernel.SetArgument(15, static_cast<int>(a_conjugate));
  kernel.SetArgument(16, static_cast<int>(b_conjugate));

  // Computes the global and local thread sizes
  const auto m_ceiled = Ceil(m, db_["WGD"]);
  const auto n_ceiled = Ceil(n, db_["WGD"]);
  const auto global =
      std::vector<size_t>{//  CeilDiv(m * db_["MDIMCD"], db_["WGD"]),
                          //  CeilDiv(n * db_["NDIMCD"], db_["WGD"])
                          (m_ceiled * db_["MDIMCD"]) / db_["WGD"], (n_ceiled * db_["NDIMCD"]) / db_["WGD"]};
  const auto local = std::vector<size_t>{db_["MDIMCD"], db_["NDIMCD"]};

  // Launches the kernel
  RunKernel(kernel, queue_, device_, global, local, event_);
}

// =================================================================================================

// Compiles the templated class
template class Xgemm<half>;
template class Xgemm<float>;
template class Xgemm<double>;
template class Xgemm<float2>;
template class Xgemm<double2>;

// =================================================================================================
}  // namespace clblast
