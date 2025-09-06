
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xgemm class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level3/xgemm.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "clblast.h"
#include "routine.hpp"
#include "routines/common.hpp"
#include "utilities/backend.hpp"
#include "utilities/buffer_test.hpp"
#include "utilities/clblast_exceptions.hpp"
#include "utilities/utilities.hpp"

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

// The main routine
template <typename T>
void Xgemm<T>::DoGemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose, const size_t m,
                      const size_t n, const size_t k, const T alpha, const Buffer<T>& a_buffer, const size_t a_offset,
                      const size_t a_ld, const Buffer<T>& b_buffer, const size_t b_offset, const size_t b_ld,
                      const T beta, const Buffer<T>& c_buffer, const size_t c_offset, const size_t c_ld,
                      const Buffer<T>& temp_buffer, const bool temp_buffer_provided) {  // optional arguments

  // Two methods to choose from, select which one to run
  const auto do_gemm_direct = UseDirectKernel(m, n, k, getDatabase()["XGEMM_MIN_INDIRECT_SIZE"]);
  const auto gemm_kernel_id = (do_gemm_direct) ? 0 : getDatabase()["GEMMK"];

  // Computes the transpose/conjugate options and sets the a/b/c sizes based on that
  bool a_do_transpose = false;
  bool b_do_transpose = false;
  bool c_do_transpose = false;
  bool a_conjugate = false;
  bool b_conjugate = false;
  size_t a_one = 0;
  size_t a_two = 0;
  size_t b_one = 0;
  size_t b_two = 0;
  size_t c_one = 0;
  size_t c_two = 0;
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
  const auto global_divider_one = c_want_rotated_(getDatabase()["GEMMK"]) ? getDatabase()["NWG"] : getDatabase()["MWG"];
  const auto global_divider_two = c_want_rotated_(getDatabase()["GEMMK"]) ? getDatabase()["MWG"] : getDatabase()["NWG"];
  const auto m_ceiled = Ceil(m, global_divider_one);
  const auto n_ceiled = Ceil(n, global_divider_two);
  const auto k_ceiled = Ceil(k, getDatabase()["KWG"] * getDatabase()["KREG"]);

  // Computes the first and second "internal" (ceiled) dimensions of the 3 matrices taking into account
  // whether the matrices need to be rotated or not for the kernel.
  size_t a_one_i = 0;
  size_t a_two_i = 0;
  size_t b_one_i = 0;
  size_t b_two_i = 0;
  size_t c_one_i = 0;
  size_t c_two_i = 0;
  CalculateInternalDimensions(m, n, k, getDatabase()["MWG"], getDatabase()["NWG"],
                              getDatabase()["KWG"] * getDatabase()["KREG"], a_one_i, a_two_i, b_one_i, b_two_i, c_one_i,
                              c_two_i, getDatabase()["GEMMK"]);

  // Determines whether or not temporary matrices are needed
  auto a_no_temp = NoTempBuffer(a_one, a_one_i, a_two, a_two_i, a_ld, a_offset, a_do_transpose, a_conjugate);
  auto b_no_temp = NoTempBuffer(b_one, b_one_i, b_two, b_two_i, b_ld, b_offset, b_do_transpose, b_conjugate);
  auto c_no_temp = NoTempBuffer(c_one, c_one_i, c_two, c_two_i, c_ld, c_offset, c_do_transpose, false);

  // Computes the sizes and offsets for (optional) temporary buffers for the 3 matrices
  auto b_temp_offset = size_t{0};
  auto c_temp_offset = size_t{0};
  const auto temp_size = ComputeTempSize(a_no_temp, b_no_temp, c_no_temp, a_one_i * a_two_i, b_one_i * b_two_i,
                                         c_one_i * c_two_i, b_temp_offset, c_temp_offset);
  if (!IsMultiple(b_temp_offset, getDatabase()["VWN"])) {
    throw BLASError(StatusCode::kUnexpectedError);
  }
  if (!IsMultiple(c_temp_offset, getDatabase()["VWM"])) {
    throw BLASError(StatusCode::kUnexpectedError);
  }

  // Creates the buffer for the (optional) temporary matrices. Note that we use 'a_buffer' in case
  // when no temporary buffer is needed, but that's just to make it compile: it is never used.
  Buffer<T> temp_buffer_all{nullptr};
  if (temp_buffer_provided) {
    temp_buffer_all = temp_buffer;
  } else {
    if (temp_size > 0) {
      temp_buffer_all = Buffer<T>(getContext(), temp_size);
    } else {
      temp_buffer_all = a_buffer;
    }
  }

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
    PadCopyTransposeMatrix(getQueue(), getDevice(), getDatabase(), eventProcessA.pointer(), emptyEventList, a_one,
                           a_two, a_ld, a_offset, a_buffer, a_one_i, a_two_i, a_one_i, 0, a_temp, ConstantOne<T>(),
                           getProgram(), true, a_do_transpose, a_conjugate);
    eventWaitList.push_back(eventProcessA);
  }

  // As above, but now for matrix B
  if (!b_no_temp) {
    auto eventProcessB = Event();
    PadCopyTransposeMatrix(getQueue(), getDevice(), getDatabase(), eventProcessB.pointer(), emptyEventList, b_one,
                           b_two, b_ld, b_offset, b_buffer, b_one_i, b_two_i, b_one_i, b_temp_offset, b_temp,
                           ConstantOne<T>(), getProgram(), true, b_do_transpose, b_conjugate);
    eventWaitList.push_back(eventProcessB);
  }

  // As above, but now for matrix C. This is only necessary if C is used both as input and output.
  if (!c_no_temp && beta != static_cast<T>(0)) {
    auto eventProcessC = Event();
    PadCopyTransposeMatrix(getQueue(), getDevice(), getDatabase(), eventProcessC.pointer(), emptyEventList, c_one,
                           c_two, c_ld, c_offset, c_buffer, c_one_i, c_two_i, c_one_i, c_temp_offset, c_temp,
                           ConstantOne<T>(), getProgram(), true, c_do_transpose, false);
    eventWaitList.push_back(eventProcessC);
  }

  // Retrieves the Xgemm kernel from the compiled binary
  auto kernel = Kernel(getProgram(), "Xgemm");

  // Sets the kernel arguments
  kernel.SetArgument(0, static_cast<int>(m_ceiled));
  kernel.SetArgument(1, static_cast<int>(n_ceiled));
  kernel.SetArgument(2, static_cast<int>(k_ceiled));
  kernel.SetArgument(3, GetRealArg(alpha));
  kernel.SetArgument(4, GetRealArg(beta));
  kernel.SetArgument(5, a_temp());
  kernel.SetArgument(6, b_temp());
  kernel.SetArgument(7, c_temp());
  kernel.SetArgument(8, static_cast<int>(b_temp_offset / getDatabase()["VWN"]));
  kernel.SetArgument(9, static_cast<int>(c_temp_offset / getDatabase()["VWM"]));

  // Computes the global and local thread sizes
  const auto global = std::vector<size_t>{(c_one_i * getDatabase()["MDIMC"]) / getDatabase()["MWG"],
                                          (c_two_i * getDatabase()["NDIMC"]) / getDatabase()["NWG"]};
  const auto local = std::vector<size_t>{getDatabase()["MDIMC"], getDatabase()["NDIMC"]};

  // Launches the kernel
  auto eventKernel = Event();
  auto eventPointer = (!c_no_temp) ? eventKernel.pointer() : getEvent();
  RunKernel(kernel, getQueue(), getDevice(), global, local, eventPointer, eventWaitList);

  // Runs the post-processing kernel if needed
  if (!c_no_temp) {
    eventWaitList.push_back(eventKernel);
    PadCopyTransposeMatrix(getQueue(), getDevice(), getDatabase(), getEvent(), eventWaitList, c_one_i, c_two_i, c_one_i,
                           c_temp_offset, c_temp, c_one, c_two, c_ld, c_offset, c_buffer, ConstantOne<T>(),
                           getProgram(), false, c_do_transpose, false);
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
  const char* name = nullptr;
  if (a_do_transpose) {
    if (b_do_transpose) {
      name = "XgemmDirectTT";
    } else {
      name = "XgemmDirectTN";
    }
  } else {
    if (b_do_transpose) {
      name = "XgemmDirectNT";
    } else {
      name = "XgemmDirectNN";
    }
  }
  auto kernel = Kernel(getProgram(), name);

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
  const auto m_ceiled = Ceil(m, getDatabase()["WGD"]);
  const auto n_ceiled = Ceil(n, getDatabase()["WGD"]);
  const auto global = std::vector<size_t>{//  CeilDiv(m * getDatabase()["MDIMCD"], getDatabase()["WGD"]),
                                          //  CeilDiv(n * getDatabase()["NDIMCD"], getDatabase()["WGD"])
                                          (m_ceiled * getDatabase()["MDIMCD"]) / getDatabase()["WGD"],
                                          (n_ceiled * getDatabase()["NDIMCD"]) / getDatabase()["WGD"]};
  const auto local = std::vector<size_t>{getDatabase()["MDIMCD"], getDatabase()["NDIMCD"]};

  // Launches the kernel
  RunKernel(kernel, getQueue(), getDevice(), global, local, getEvent());
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
