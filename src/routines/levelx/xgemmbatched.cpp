
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
#include "routines/level3/xgemm.hpp"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
XgemmBatched<T>::XgemmBatched(Queue &queue, EventPointer event, const std::string &name):
    Routine(queue, event, name,
            {"Copy","Pad","Transpose","Padtranspose","Xgemm","XgemmDirect","GemmRoutine"},
            PrecisionValue<T>(), {}, {
    #include "../../kernels/level3/level3.opencl"
    #include "../../kernels/level3/copy_fast.opencl"
    #include "../../kernels/level3/copy_pad.opencl"
    #include "../../kernels/level3/transpose_fast.opencl"
    #include "../../kernels/level3/transpose_pad.opencl"
    , // separated in multiple parts to prevent C1091 in MSVC 2013
    #include "../../kernels/level3/xgemm_direct_part1.opencl"
    #include "../../kernels/level3/xgemm_direct_part2.opencl"
    #include "../../kernels/level3/xgemm_direct_part3.opencl"
    , // separated in multiple parts to prevent C1091 in MSVC 2013
    #include "../../kernels/level3/xgemm_part1.opencl"
    #include "../../kernels/level3/xgemm_part2.opencl"
    , // separated in multiple parts to prevent C1091 in MSVC 2013
    #include "../../kernels/level3/xgemm_part3.opencl"
    #include "../../kernels/level3/xgemm_part4.opencl"
    , // separated in multiple parts to prevent C1091 in MSVC 2013
    #include "../../kernels/level3/xgemm_batched.opencl"
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

  // Two methods to choose from, select which one to run
  const auto do_gemm_direct = Xgemm<T>::UseDirectKernel(m, n, k, db_["XGEMM_MIN_INDIRECT_SIZE"]);
  const auto gemm_kernel_id = (do_gemm_direct) ? 0 : db_["GEMMK"];

  // Computes the transpose/conjugate options and sets the a/b/c sizes based on that
  bool a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate;
  size_t a_one, a_two, b_one, b_two, c_one, c_two;
  Xgemm<T>::ProcessArguments(layout, a_transpose, b_transpose, m, n, k,
                             a_one, a_two, b_one, b_two, c_one, c_two,
                             a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate,
                             gemm_kernel_id);

  // Tests the matrices for validity
  for (auto batch = size_t{0}; batch < batch_count; ++batch) {
    TestMatrixA(a_one, a_two, a_buffer, a_offsets[batch], a_ld, false); // don't test for invalid LD
    TestMatrixB(b_one, b_two, b_buffer, b_offsets[batch], b_ld, false); // don't test for invalid LD
    TestMatrixC(c_one, c_two, c_buffer, c_offsets[batch], c_ld);
  }

  // Upload the scalar arguments to the device
  auto alphas_device = Buffer<T>(context_, BufferAccess::kReadWrite, batch_count);
  auto betas_device = Buffer<T>(context_, BufferAccess::kReadWrite, batch_count);
  alphas_device.Write(queue_, batch_count, alphas);
  betas_device.Write(queue_, batch_count, betas);

  // Converts the offset to integers
  auto a_offsets_int = std::vector<int>(batch_count);
  auto b_offsets_int = std::vector<int>(batch_count);
  auto c_offsets_int = std::vector<int>(batch_count);
  for (auto batch = size_t{ 0 }; batch < batch_count; ++batch) {
    a_offsets_int[batch] = static_cast<int>(a_offsets[batch]);
    b_offsets_int[batch] = static_cast<int>(b_offsets[batch]);
    c_offsets_int[batch] = static_cast<int>(c_offsets[batch]);
  }

  // Selects which version of the batched GEMM to run
  if (do_gemm_direct) { // single generic kernel
    BatchedGemmDirect(m, n, k, alphas_device,
                      a_buffer, a_offsets_int, a_ld, b_buffer, b_offsets_int, b_ld,
                      betas_device, c_buffer, c_offsets_int, c_ld,
                      a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate,
                      batch_count);
  }
  else { // pre/post-processing plus a very fast kernel
    BatchedGemmIndirect(m, n, k, alphas_device,
                        a_buffer, a_offsets_int, a_ld, b_buffer, b_offsets_int, b_ld,
                        betas_device, c_buffer, c_offsets_int, c_ld,
                        a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate,
                        a_one, a_two, b_one, b_two, c_one, c_two, batch_count);
  }
}


// =================================================================================================

// The indirect version of batched GEMM. This uses the faster but non-general kernel. It has specific
// requirements, but several pre and post-processing kernels take care of those. However, the
// overhead of these extra kernels might not be ideal for certain devices/arguments.
template <typename T>
void XgemmBatched<T>::BatchedGemmIndirect(const size_t m, const size_t n, const size_t k,
                                          const Buffer<T> &alphas,
                                          const Buffer<T> &a_buffer, const std::vector<int> &a_offsets, const size_t a_ld,
                                          const Buffer<T> &b_buffer, const std::vector<int> &b_offsets, const size_t b_ld,
                                          const Buffer<T> &betas,
                                          const Buffer<T> &c_buffer, const std::vector<int> &c_offsets, const size_t c_ld,
                                          const bool a_do_transpose, const bool b_do_transpose, const bool c_do_transpose,
                                          const bool a_conjugate, const bool b_conjugate,
                                          const size_t a_one, const size_t a_two,
                                          const size_t b_one, const size_t b_two,
                                          const size_t c_one, const size_t c_two,
                                          const size_t batch_count) {
  // Calculates the ceiled versions of m, n, and k
  const auto m_ceiled = Ceil(Ceil(m, db_["MWG"]), db_["VWM"]);
  const auto n_ceiled = Ceil(Ceil(n, db_["NWG"]), db_["VWN"]);
  const auto k_ceiled = Ceil(Ceil(k, db_["KWG"]), db_["VWM"]);

  // Computes the first and second "internal" (ceiled) dimensions of the 3 matrices taking into account
  // whether the matrices need to be rotated or not for the kernel.
  size_t a_one_i, a_two_i, b_one_i, b_two_i, c_one_i, c_two_i;
  Xgemm<T>::CalculateInternalDimensions(m, n, k, db_["MWG"], db_["NWG"], db_["KWG"],
                                        a_one_i, a_two_i, b_one_i, b_two_i, c_one_i, c_two_i,
                                        db_["GEMMK"]);

  // Sets the "internal" offsets, i.e. the perfect offsets
  auto a_offsets_i = std::vector<int>(batch_count);
  auto b_offsets_i = std::vector<int>(batch_count);
  auto c_offsets_i = std::vector<int>(batch_count);
  for (auto batch = size_t{0}; batch < batch_count; ++batch) {
    a_offsets_i[batch] = static_cast<int>(batch * a_one_i * a_two_i);
    b_offsets_i[batch] = static_cast<int>(batch * b_one_i * b_two_i);
    c_offsets_i[batch] = static_cast<int>(batch * c_one_i * c_two_i);
  }

  // Determines whether or not temporary matrices are needed
  auto a_no_temp = a_one == a_one_i && a_two == a_two_i && a_ld == a_one && a_offsets == a_offsets_i &&
                   !a_do_transpose && !a_conjugate;
  auto b_no_temp = b_one == b_one_i && b_two == b_two_i && b_ld == b_one && b_offsets == b_offsets_i &&
                   !b_do_transpose && !b_conjugate;
  auto c_no_temp = c_one == c_one_i && c_two == c_two_i && c_ld == c_one && c_offsets == c_offsets_i &&
                   !c_do_transpose;

  // Creates the temporary matrices
  const auto a_temp = (a_no_temp) ? a_buffer : Buffer<T>(context_, batch_count * a_one_i * a_two_i);
  const auto b_temp = (b_no_temp) ? b_buffer : Buffer<T>(context_, batch_count * b_one_i * b_two_i);
  const auto c_temp = (c_no_temp) ? c_buffer : Buffer<T>(context_, batch_count * c_one_i * c_two_i);

  // Events of all kernels (including pre/post processing kernels)
  auto eventWaitList = std::vector<Event>();
  auto emptyEventList = std::vector<Event>();

  // Runs the pre-processing kernel for matrix A. This transposes the matrix, but also pads zeros
  // to fill it up until it reaches a certain multiple of size (kernel parameter dependent). In
  // case nothing has to be done, these kernels can be skipped.
  if (!a_no_temp) {
    auto a_offsets_device = Buffer<int>(context_, BufferAccess::kReadWrite, batch_count);
    auto a_offsets_i_device = Buffer<int>(context_, BufferAccess::kReadWrite, batch_count);
    a_offsets_device.Write(queue_, batch_count, a_offsets);
    a_offsets_i_device.Write(queue_, batch_count, a_offsets_i);
    auto eventProcessA = Event();
    PadCopyTransposeMatrixBatched(queue_, device_, db_, eventProcessA.pointer(), emptyEventList,
                                  a_one, a_two, a_ld, a_offsets_device, a_buffer,
                                  a_one_i, a_two_i, a_one_i, a_offsets_i_device, a_temp,
                                  program_, true, a_do_transpose, a_conjugate, batch_count);
    eventWaitList.push_back(eventProcessA);
  }

  // As above, but now for matrix B
  if (!b_no_temp) {
    auto b_offsets_device = Buffer<int>(context_, BufferAccess::kReadWrite, batch_count);
    auto b_offsets_i_device = Buffer<int>(context_, BufferAccess::kReadWrite, batch_count);
    b_offsets_device.Write(queue_, batch_count, b_offsets);
    b_offsets_i_device.Write(queue_, batch_count, b_offsets_i);
    auto eventProcessB = Event();
    PadCopyTransposeMatrixBatched(queue_, device_, db_, eventProcessB.pointer(), emptyEventList,
                                  b_one, b_two, b_ld, b_offsets_device, b_buffer,
                                  b_one_i, b_two_i, b_one_i, b_offsets_i_device, b_temp,
                                  program_, true, b_do_transpose, b_conjugate, batch_count);
    eventWaitList.push_back(eventProcessB);
  }

  // As above, but now for matrix C
  auto c_offsets_device = Buffer<int>(context_, BufferAccess::kReadWrite, batch_count);
  auto c_offsets_i_device = Buffer<int>(context_, BufferAccess::kReadWrite, batch_count);
  if (!c_no_temp) {
    c_offsets_device.Write(queue_, batch_count, c_offsets);
    c_offsets_i_device.Write(queue_, batch_count, c_offsets_i);
    auto eventProcessC = Event();
    PadCopyTransposeMatrixBatched(queue_, device_, db_, eventProcessC.pointer(), emptyEventList,
                                  c_one, c_two, c_ld, c_offsets_device, c_buffer,
                                  c_one_i, c_two_i, c_one_i, c_offsets_i_device, c_temp,
                                  program_, true, c_do_transpose, false, batch_count);
    eventWaitList.push_back(eventProcessC);
  }

  // Retrieves the Xgemm kernel from the compiled binary
  auto kernel = Kernel(program_, "XgemmBatched");

  // Sets the kernel arguments
  kernel.SetArgument(0, static_cast<int>(m_ceiled));
  kernel.SetArgument(1, static_cast<int>(n_ceiled));
  kernel.SetArgument(2, static_cast<int>(k_ceiled));
  kernel.SetArgument(3, alphas());
  kernel.SetArgument(4, betas());
  kernel.SetArgument(5, a_temp());
  kernel.SetArgument(6, static_cast<int>(a_one_i));
  kernel.SetArgument(7, static_cast<int>(a_two_i));
  kernel.SetArgument(8, b_temp());
  kernel.SetArgument(9, static_cast<int>(b_one_i));
  kernel.SetArgument(10, static_cast<int>(b_two_i));
  kernel.SetArgument(11, c_temp());
  kernel.SetArgument(12, static_cast<int>(c_one_i));
  kernel.SetArgument(13, static_cast<int>(c_two_i));

  // Computes the global and local thread sizes
  const auto global = std::vector<size_t>{
    (c_one_i * db_["MDIMC"]) / db_["MWG"],
    (c_two_i * db_["NDIMC"]) / db_["NWG"],
    batch_count
  };
  const auto local = std::vector<size_t>{db_["MDIMC"], db_["NDIMC"], 1};

  // Launches the kernel
  auto eventKernel = Event();
  auto eventPointer = (!c_no_temp) ? eventKernel.pointer() : event_;
  RunKernel(kernel, queue_, device_, global, local, eventPointer, eventWaitList);

  // Runs the post-processing kernel if needed
  if (!c_no_temp) {
    eventWaitList.push_back(eventKernel);
    PadCopyTransposeMatrixBatched(queue_, device_, db_, event_, eventWaitList,
                                  c_one_i, c_two_i, c_one_i, c_offsets_i_device, c_temp,
                                  c_one, c_two, c_ld, c_offsets_device, c_buffer,
                                  program_, false, c_do_transpose, false, batch_count);
  }
}

// =================================================================================================

// The direct version of batched GEMM, requiring just one kernel, no pre or post-processing kernels.
template <typename T>
void XgemmBatched<T>::BatchedGemmDirect(const size_t m, const size_t n, const size_t k,
                                        const Buffer<T> &alphas,
                                        const Buffer<T> &a_buffer, const std::vector<int> &a_offsets, const size_t a_ld,
                                        const Buffer<T> &b_buffer, const std::vector<int> &b_offsets, const size_t b_ld,
                                        const Buffer<T> &betas,
                                        const Buffer<T> &c_buffer, const std::vector<int> &c_offsets, const size_t c_ld,
                                        const bool a_do_transpose, const bool b_do_transpose, const bool c_do_transpose,
                                        const bool a_conjugate, const bool b_conjugate,
                                        const size_t batch_count) {

  // Uploads the offsets to the device
  auto a_offsets_device = Buffer<int>(context_, BufferAccess::kReadWrite, batch_count);
  auto b_offsets_device = Buffer<int>(context_, BufferAccess::kReadWrite, batch_count);
  auto c_offsets_device = Buffer<int>(context_, BufferAccess::kReadWrite, batch_count);
  a_offsets_device.Write(queue_, batch_count, a_offsets);
  b_offsets_device.Write(queue_, batch_count, b_offsets);
  c_offsets_device.Write(queue_, batch_count, c_offsets);

  // Retrieves the proper XgemmDirect kernel from the compiled binary
  const auto name = (a_do_transpose) ? (b_do_transpose ? "XgemmDirectBatchedTT" : "XgemmDirectBatchedTN") :
                                       (b_do_transpose ? "XgemmDirectBatchedNT" : "XgemmDirectBatchedNN");
  auto kernel = Kernel(program_, name);

  // Sets the kernel arguments
  kernel.SetArgument(0, static_cast<int>(m));
  kernel.SetArgument(1, static_cast<int>(n));
  kernel.SetArgument(2, static_cast<int>(k));
  kernel.SetArgument(3, alphas());
  kernel.SetArgument(4, betas());
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
