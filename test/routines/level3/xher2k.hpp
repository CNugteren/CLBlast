
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements a class with static methods to describe the Xher2k routine. Examples of
// such 'descriptions' are how to calculate the size a of buffer or how to run the routine. These
// static methods are used by the correctness tester and the performance tester.
//
// =================================================================================================

#ifndef CLBLAST_TEST_ROUTINES_XHER2K_H_
#define CLBLAST_TEST_ROUTINES_XHER2K_H_

#include "test/routines/common.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T, typename U>
class TestXher2k {
 public:

  // The BLAS level: 1, 2, or 3
  static size_t BLASLevel() { return 3; }

  // The list of arguments relevant for this routine
  static std::vector<std::string> GetOptions() {
    return {kArgN, kArgK,
            kArgLayout, kArgTriangle, kArgATransp,
            kArgALeadDim, kArgBLeadDim, kArgCLeadDim,
            kArgAOffset, kArgBOffset, kArgCOffset,
            kArgAlpha, kArgBeta};
  }
  static std::vector<std::string> BuffersIn() { return {kBufMatA, kBufMatB, kBufMatC}; }
  static std::vector<std::string> BuffersOut() { return {kBufMatC}; }

  // Describes how to obtain the sizes of the buffers
  static size_t GetSizeA(const Arguments<U> &args) {
    auto a_rotated = (args.layout == Layout::kColMajor && args.a_transpose != Transpose::kNo) ||
                     (args.layout == Layout::kRowMajor && args.a_transpose == Transpose::kNo);
    auto a_two = (a_rotated) ? args.n : args.k;
    return a_two * args.a_ld + args.a_offset;
  }
  static size_t GetSizeB(const Arguments<U> &args) {
    auto b_rotated = (args.layout == Layout::kColMajor && args.a_transpose != Transpose::kNo) ||
                     (args.layout == Layout::kRowMajor && args.a_transpose == Transpose::kNo);
    auto b_two = (b_rotated) ? args.n : args.k;
    return b_two * args.b_ld + args.b_offset;
  }
  static size_t GetSizeC(const Arguments<U> &args) {
    return args.n * args.c_ld + args.c_offset;
  }

  // Describes how to set the sizes of all the buffers
  static void SetSizes(Arguments<U> &args, Queue&) {
    args.a_size = GetSizeA(args);
    args.b_size = GetSizeB(args);
    args.c_size = GetSizeC(args);
  }

  // Describes what the default values of the leading dimensions of the matrices are
  static size_t DefaultLDA(const Arguments<U> &args) { return args.k; }
  static size_t DefaultLDB(const Arguments<U> &args) { return args.k; }
  static size_t DefaultLDC(const Arguments<U> &args) { return args.n; }

  // Describes which transpose options are relevant for this routine
  using Transposes = std::vector<Transpose>;
  static Transposes GetATransposes(const Transposes &) { return {Transpose::kNo, Transpose::kConjugate}; }
  static Transposes GetBTransposes(const Transposes &) { return {}; } // N/A for this routine

  // Describes how to prepare the input data
  static void PrepareData(const Arguments<U>&, Queue&, const int, std::vector<T>&,
                          std::vector<T>&, std::vector<T>&, std::vector<T>&, std::vector<T>&,
                          std::vector<T>&, std::vector<T>&) {} // N/A for this routine

  // Describes how to run the CLBlast routine
  static StatusCode RunRoutine(const Arguments<U> &args, Buffers<T> &buffers, Queue &queue) {
    auto alpha2 = T{args.alpha, args.alpha};
    #ifdef OPENCL_API
      auto queue_plain = queue();
      auto event = cl_event{};
      auto status = Her2k(args.layout, args.triangle, args.a_transpose,
                          args.n, args.k, alpha2,
                          buffers.a_mat(), args.a_offset, args.a_ld,
                          buffers.b_mat(), args.b_offset, args.b_ld, args.beta,
                          buffers.c_mat(), args.c_offset, args.c_ld,
                          &queue_plain, &event);
      if (status == StatusCode::kSuccess) { clWaitForEvents(1, &event); clReleaseEvent(event); }
    #elif CUDA_API
      auto status = Her2k(args.layout, args.triangle, args.a_transpose,
                          args.n, args.k, alpha2,
                          buffers.a_mat(), args.a_offset, args.a_ld,
                          buffers.b_mat(), args.b_offset, args.b_ld, args.beta,
                          buffers.c_mat(), args.c_offset, args.c_ld,
                          queue.GetContext()(), queue.GetDevice()());
      cuStreamSynchronize(queue());
    #endif
    return status;
  }

  // Describes how to run the clBLAS routine (for correctness/performance comparison)
  #ifdef CLBLAST_REF_CLBLAS
    static StatusCode RunReference1(const Arguments<U> &args, Buffers<T> &buffers, Queue &queue) {
      auto queue_plain = queue();
      auto event = cl_event{};
      auto alpha2 = T{args.alpha, args.alpha};
      auto status = clblasXher2k(convertToCLBLAS(args.layout),
                                 convertToCLBLAS(args.triangle),
                                 convertToCLBLAS(args.a_transpose),
                                 args.n, args.k, alpha2,
                                 buffers.a_mat, args.a_offset, args.a_ld,
                                 buffers.b_mat, args.b_offset, args.b_ld, args.beta,
                                 buffers.c_mat, args.c_offset, args.c_ld,
                                 1, &queue_plain, 0, nullptr, &event);
      clWaitForEvents(1, &event);
      return static_cast<StatusCode>(status);
    }
  #endif

  // Describes how to run the CPU BLAS routine (for correctness/performance comparison)
  #ifdef CLBLAST_REF_CBLAS
    static StatusCode RunReference2(const Arguments<U> &args, BuffersHost<T> &buffers_host, Queue&) {
      auto alpha2 = T{args.alpha, args.alpha};
      cblasXher2k(convertToCBLAS(args.layout),
                  convertToCBLAS(args.triangle),
                  convertToCBLAS(args.a_transpose),
                  args.n, args.k, alpha2,
                  buffers_host.a_mat, args.a_offset, args.a_ld,
                  buffers_host.b_mat, args.b_offset, args.b_ld, args.beta,
                  buffers_host.c_mat, args.c_offset, args.c_ld);
      return StatusCode::kSuccess;
    }
  #endif

  // Describes how to run the cuBLAS routine (for correctness/performance comparison)
  #ifdef CLBLAST_REF_CUBLAS
    static StatusCode RunReference3(const Arguments<U> &args, BuffersCUDA<T> &buffers, Queue &) {
      auto alpha2 = T{args.alpha, args.alpha};
      auto status = cublasXher2k(reinterpret_cast<cublasHandle_t>(args.cublas_handle), args.layout,
                                 convertToCUBLAS(args.triangle),
                                 convertToCUBLAS(args.a_transpose),
                                 args.n, args.k, alpha2,
                                 buffers.a_mat, args.a_offset, args.a_ld,
                                 buffers.b_mat, args.b_offset, args.b_ld, args.beta,
                                 buffers.c_mat, args.c_offset, args.c_ld);
      if (status == CUBLAS_STATUS_SUCCESS) { return StatusCode::kSuccess; } else { return StatusCode::kUnknownError; }
    }
  #endif

  // Describes how to download the results of the computation (more importantly: which buffer)
  static std::vector<T> DownloadResult(const Arguments<U> &args, Buffers<T> &buffers, Queue &queue) {
    std::vector<T> result(args.c_size, static_cast<T>(0));
    buffers.c_mat.Read(queue, args.c_size, result);
    return result;
  }

  // Describes how to compute the indices of the result buffer
  static size_t ResultID1(const Arguments<U> &args) { return args.n; }
  static size_t ResultID2(const Arguments<U> &args) { return args.n; }
  static size_t GetResultIndex(const Arguments<U> &args, const size_t id1, const size_t id2) {
    return id1*args.c_ld + id2 + args.c_offset;
  }

  // Describes how to compute performance metrics
  static size_t GetFlops(const Arguments<U> &args) {
    return 2 * args.n * args.n * args.k;
  }
  static size_t GetBytes(const Arguments<U> &args) {
    return (args.n*args.k + args.n*args.n) * sizeof(T);
  }
};

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_ROUTINES_XHER2K_H_
#endif
