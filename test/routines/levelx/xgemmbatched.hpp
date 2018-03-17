
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements a class with static methods to describe the XgemmBatched routine. Examples of
// such 'descriptions' are how to calculate the size a of buffer or how to run the routine. These
// static methods are used by the correctness tester and the performance tester.
//
// =================================================================================================

#ifndef CLBLAST_TEST_ROUTINES_XGEMMBATCHED_H_
#define CLBLAST_TEST_ROUTINES_XGEMMBATCHED_H_

#include "test/routines/common.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class TestXgemmBatched {
 public:

  // Although it is a non-BLAS routine, it can still be tested against level-3 routines in a loop
  static size_t BLASLevel() { return 3; }

  // The list of arguments relevant for this routine
  static std::vector<std::string> GetOptions() {
    return {kArgM, kArgN, kArgK,
            kArgLayout, kArgATransp, kArgBTransp,
            kArgALeadDim, kArgBLeadDim, kArgCLeadDim,
            kArgAOffset, kArgBOffset, kArgCOffset,
            kArgBatchCount, kArgAlpha, kArgBeta};
  }
  static std::vector<std::string> BuffersIn() { return {kBufMatA, kBufMatB, kBufMatC}; }
  static std::vector<std::string> BuffersOut() { return {kBufMatC}; }

  // Helper for the sizes per batch
  static size_t PerBatchSizeA(const Arguments<T> &args) {
    auto a_rotated = (args.layout == Layout::kColMajor && args.a_transpose != Transpose::kNo) ||
                     (args.layout == Layout::kRowMajor && args.a_transpose == Transpose::kNo);
    auto a_two = (a_rotated) ? args.m : args.k;
    return a_two * args.a_ld;
  }
  static size_t PerBatchSizeB(const Arguments<T> &args) {
    auto b_rotated = (args.layout == Layout::kColMajor && args.b_transpose != Transpose::kNo) ||
                     (args.layout == Layout::kRowMajor && args.b_transpose == Transpose::kNo);
    auto b_two = (b_rotated) ? args.k : args.n;
    return b_two * args.b_ld;
  }
  static size_t PerBatchSizeC(const Arguments<T> &args) {
    auto c_rotated = (args.layout == Layout::kRowMajor);
    auto c_two = (c_rotated) ? args.m : args.n;
    return c_two * args.c_ld;
  }

  // Describes how to obtain the sizes of the buffers
  static size_t GetSizeA(const Arguments<T> &args) {
    return PerBatchSizeA(args) * args.batch_count + args.a_offset;
  }
  static size_t GetSizeB(const Arguments<T> &args) {
    return PerBatchSizeB(args) * args.batch_count + args.b_offset;
  }
  static size_t GetSizeC(const Arguments<T> &args) {
    return PerBatchSizeC(args) * args.batch_count + args.c_offset;
  }

  // Describes how to set the sizes of all the buffers
  static void SetSizes(Arguments<T> &args, Queue&) {
    args.a_size = GetSizeA(args);
    args.b_size = GetSizeB(args);
    args.c_size = GetSizeC(args);

    // Also sets the batch-related variables
    args.a_offsets = std::vector<size_t>(args.batch_count);
    args.b_offsets = std::vector<size_t>(args.batch_count);
    args.c_offsets = std::vector<size_t>(args.batch_count);
    args.alphas = std::vector<T>(args.batch_count);
    args.betas = std::vector<T>(args.batch_count);
    for (auto batch = size_t{0}; batch < args.batch_count; ++batch) {
      args.a_offsets[batch] = batch * PerBatchSizeA(args) + args.a_offset;
      args.b_offsets[batch] = batch * PerBatchSizeB(args) + args.b_offset;
      args.c_offsets[batch] = batch * PerBatchSizeC(args) + args.c_offset;
      args.alphas[batch] = args.alpha + Constant<T>(static_cast<double>(batch + 1));
      args.betas[batch] = args.beta + Constant<T>(static_cast<double>(batch + 1));
    }
  }

  // Describes what the default values of the leading dimensions of the matrices are
  static size_t DefaultLDA(const Arguments<T> &args) { return args.k; }
  static size_t DefaultLDB(const Arguments<T> &args) { return args.n; }
  static size_t DefaultLDC(const Arguments<T> &args) { return args.n; }

  // Describes which transpose options are relevant for this routine
  using Transposes = std::vector<Transpose>;
  static Transposes GetATransposes(const Transposes &all) { return all; }
  static Transposes GetBTransposes(const Transposes &all) { return all; }

  // Describes how to prepare the input data
  static void PrepareData(const Arguments<T>&, Queue&, const int, std::vector<T>&,
                          std::vector<T>&, std::vector<T>&, std::vector<T>&, std::vector<T>&,
                          std::vector<T>&, std::vector<T>&) {} // N/A for this routine

  // Describes how to run the CLBlast routine
  static StatusCode RunRoutine(const Arguments<T> &args, Buffers<T> &buffers, Queue &queue) {
    // Relaxed requirement on ld_a and ld_b within the library, this is here to match clBLAS
    auto a_rotated = (args.layout == Layout::kColMajor && args.a_transpose != Transpose::kNo) ||
                     (args.layout == Layout::kRowMajor && args.a_transpose == Transpose::kNo);
    auto b_rotated = (args.layout == Layout::kColMajor && args.b_transpose != Transpose::kNo) ||
                     (args.layout == Layout::kRowMajor && args.b_transpose == Transpose::kNo);
    auto a_one = (!a_rotated) ? args.m : args.k;
    auto b_one = (!b_rotated) ? args.k : args.n;
    if (args.a_ld < a_one) { return StatusCode::kInvalidLeadDimA; }
    if (args.b_ld < b_one) { return StatusCode::kInvalidLeadDimB; }
    #ifdef OPENCL_API
      auto queue_plain = queue();
      auto event = cl_event{};
      auto status = GemmBatched(args.layout, args.a_transpose, args.b_transpose,
                                args.m, args.n, args.k, args.alphas.data(),
                                buffers.a_mat(), args.a_offsets.data(), args.a_ld,
                                buffers.b_mat(), args.b_offsets.data(), args.b_ld, args.betas.data(),
                                buffers.c_mat(), args.c_offsets.data(), args.c_ld,
                                args.batch_count,
                                &queue_plain, &event);
      if (status == StatusCode::kSuccess) { clWaitForEvents(1, &event); clReleaseEvent(event); }
    #elif CUDA_API
      auto status = GemmBatched(args.layout, args.a_transpose, args.b_transpose,
                                args.m, args.n, args.k, args.alphas.data(),
                                buffers.a_mat(), args.a_offsets.data(), args.a_ld,
                                buffers.b_mat(), args.b_offsets.data(), args.b_ld, args.betas.data(),
                                buffers.c_mat(), args.c_offsets.data(), args.c_ld,
                                args.batch_count,
                                queue.GetContext()(), queue.GetDevice()());
      cuStreamSynchronize(queue());
    #endif
    return status;
  }

  // Describes how to run the clBLAS routine (for correctness/performance comparison)
  #ifdef CLBLAST_REF_CLBLAS
    static StatusCode RunReference1(const Arguments<T> &args, Buffers<T> &buffers, Queue &queue) {
      auto queue_plain = queue();
      for (auto batch = size_t{0}; batch < args.batch_count; ++batch) {
        auto event = cl_event{};
        auto status = clblasXgemm(convertToCLBLAS(args.layout),
                                  convertToCLBLAS(args.a_transpose),
                                  convertToCLBLAS(args.b_transpose),
                                  args.m, args.n, args.k, args.alphas[batch],
                                  buffers.a_mat, args.a_offsets[batch], args.a_ld,
                                  buffers.b_mat, args.b_offsets[batch], args.b_ld, args.betas[batch],
                                  buffers.c_mat, args.c_offsets[batch], args.c_ld,
                                  1, &queue_plain, 0, nullptr, &event);
        clWaitForEvents(1, &event);
        if (static_cast<StatusCode>(status) != StatusCode::kSuccess) {
          return static_cast<StatusCode>(status);
        }
      }
      return StatusCode::kSuccess;
    }
  #endif

  // Describes how to run the CPU BLAS routine (for correctness/performance comparison)
  #ifdef CLBLAST_REF_CBLAS
    static StatusCode RunReference2(const Arguments<T> &args, BuffersHost<T> &buffers_host, Queue &) {
      for (auto batch = size_t{0}; batch < args.batch_count; ++batch) {
        cblasXgemm(convertToCBLAS(args.layout),
                   convertToCBLAS(args.a_transpose),
                   convertToCBLAS(args.b_transpose),
                   args.m, args.n, args.k, args.alphas[batch],
                   buffers_host.a_mat, args.a_offsets[batch], args.a_ld,
                   buffers_host.b_mat, args.b_offsets[batch], args.b_ld, args.betas[batch],
                   buffers_host.c_mat, args.c_offsets[batch], args.c_ld);
      }
      return StatusCode::kSuccess;
    }
  #endif

  // Describes how to run the cuBLAS routine (for correctness/performance comparison)
  #ifdef CLBLAST_REF_CUBLAS
    static StatusCode RunReference3(const Arguments<T> &args, BuffersCUDA<T> &buffers, Queue &) {
      for (auto batch = size_t{0}; batch < args.batch_count; ++batch) {
        auto status = cublasXgemm(reinterpret_cast<cublasHandle_t>(args.cublas_handle), args.layout,
                                  convertToCUBLAS(args.a_transpose),
                                  convertToCUBLAS(args.b_transpose),
                                  args.m, args.n, args.k, args.alphas[batch],
                                  buffers.a_mat, args.a_offsets[batch], args.a_ld,
                                  buffers.b_mat, args.b_offsets[batch], args.b_ld, args.betas[batch],
                                  buffers.c_mat, args.c_offsets[batch], args.c_ld);
      if (status != CUBLAS_STATUS_SUCCESS) { return StatusCode::kUnknownError; }
      }
      return StatusCode::kSuccess;
    }
  #endif

  // Describes how to download the results of the computation (more importantly: which buffer)
  static std::vector<T> DownloadResult(const Arguments<T> &args, Buffers<T> &buffers, Queue &queue) {
    std::vector<T> result(args.c_size, static_cast<T>(0));
    buffers.c_mat.Read(queue, args.c_size, result);
    return result;
  }

  // Describes how to compute the indices of the result buffer
  static size_t ResultID1(const Arguments<T> &args) { return args.m; }
  static size_t ResultID2(const Arguments<T> &args) { return args.n * args.batch_count; }
  static size_t GetResultIndex(const Arguments<T> &args, const size_t id1, const size_t id2_3) {
    const size_t id2 = id2_3 % args.n;
    const size_t id3 = id2_3 / args.n;
    return (args.layout == Layout::kRowMajor) ?
           id1*args.c_ld + id2 + args.c_offsets[id3]:
           id2*args.c_ld + id1 + args.c_offsets[id3];
  }

  // Describes how to compute performance metrics
  static size_t GetFlops(const Arguments<T> &args) {
    return args.batch_count * (2 * args.m * args.n * args.k);
  }
  static size_t GetBytes(const Arguments<T> &args) {
    return args.batch_count * (args.m*args.k + args.k*args.n + 2*args.m*args.n) * sizeof(T);
  }
};

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_ROUTINES_XGEMMBATCHED_H_
#endif
