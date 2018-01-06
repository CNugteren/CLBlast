
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements a class with static methods to describe the Xtrmm routine. Examples of
// such 'descriptions' are how to calculate the size a of buffer or how to run the routine. These
// static methods are used by the correctness tester and the performance tester.
//
// =================================================================================================

#ifndef CLBLAST_TEST_ROUTINES_XTRMM_H_
#define CLBLAST_TEST_ROUTINES_XTRMM_H_

#include "test/routines/common.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class TestXtrmm {
 public:

  // The BLAS level: 1, 2, or 3
  static size_t BLASLevel() { return 3; }

  // The list of arguments relevant for this routine
  static std::vector<std::string> GetOptions() {
    return {kArgM, kArgN,
            kArgLayout, kArgSide, kArgTriangle, kArgATransp, kArgDiagonal,
            kArgALeadDim, kArgBLeadDim,
            kArgAOffset, kArgBOffset,
            kArgAlpha};
  }
  static std::vector<std::string> BuffersIn() { return {kBufMatA, kBufMatB}; }
  static std::vector<std::string> BuffersOut() { return {kBufMatB}; }

  // Describes how to obtain the sizes of the buffers
  static size_t GetSizeA(const Arguments<T> &args) {
    auto k = (args.side == Side::kLeft) ? args.m : args.n;
    return k * args.a_ld + args.a_offset;
  }
  static size_t GetSizeB(const Arguments<T> &args) {
    auto b_rotated = (args.layout == Layout::kRowMajor);
    auto b_two = (b_rotated) ? args.m : args.n;
    return b_two * args.b_ld + args.b_offset;
  }

  // Describes how to set the sizes of all the buffers
  static void SetSizes(Arguments<T> &args, Queue&) {
    args.a_size = GetSizeA(args);
    args.b_size = GetSizeB(args);
  }

  // Describes what the default values of the leading dimensions of the matrices are
  static size_t DefaultLDA(const Arguments<T> &args) { return args.m; }
  static size_t DefaultLDB(const Arguments<T> &args) { return args.n; }
  static size_t DefaultLDC(const Arguments<T> &) { return 1; } // N/A for this routine

  // Describes which transpose options are relevant for this routine
  using Transposes = std::vector<Transpose>;
  static Transposes GetATransposes(const Transposes &all) { return all; }
  static Transposes GetBTransposes(const Transposes &) { return {}; } // N/A for this routine

  // Describes how to prepare the input data
  static void PrepareData(const Arguments<T>&, Queue&, const int, std::vector<T>&,
                          std::vector<T>&, std::vector<T>&, std::vector<T>&, std::vector<T>&,
                          std::vector<T>&, std::vector<T>&) {} // N/A for this routine

  // Describes how to run the CLBlast routine
  static StatusCode RunRoutine(const Arguments<T> &args, Buffers<T> &buffers, Queue &queue) {
    #ifdef OPENCL_API
      auto queue_plain = queue();
      auto event = cl_event{};
      auto status = Trmm(args.layout, args.side, args.triangle, args.a_transpose, args.diagonal,
                         args.m, args.n, args.alpha,
                         buffers.a_mat(), args.a_offset, args.a_ld,
                         buffers.b_mat(), args.b_offset, args.b_ld,
                         &queue_plain, &event);
      if (status == StatusCode::kSuccess) { clWaitForEvents(1, &event); clReleaseEvent(event); }
    #elif CUDA_API
      auto status = Trmm(args.layout, args.side, args.triangle, args.a_transpose, args.diagonal,
                         args.m, args.n, args.alpha,
                         buffers.a_mat(), args.a_offset, args.a_ld,
                         buffers.b_mat(), args.b_offset, args.b_ld,
                         queue.GetContext()(), queue.GetDevice()());
      cuStreamSynchronize(queue());
    #endif
    return status;
  }

  // Describes how to run the clBLAS routine (for correctness/performance comparison)
  #ifdef CLBLAST_REF_CLBLAS
    static StatusCode RunReference1(const Arguments<T> &args, Buffers<T> &buffers, Queue &queue) {
      auto queue_plain = queue();
      auto event = cl_event{};
      auto status = clblasXtrmm(convertToCLBLAS(args.layout),
                                convertToCLBLAS(args.side),
                                convertToCLBLAS(args.triangle),
                                convertToCLBLAS(args.a_transpose),
                                convertToCLBLAS(args.diagonal),
                                args.m, args.n, args.alpha,
                                buffers.a_mat, args.a_offset, args.a_ld,
                                buffers.b_mat, args.b_offset, args.b_ld,
                                1, &queue_plain, 0, nullptr, &event);
      clWaitForEvents(1, &event);
      return static_cast<StatusCode>(status);
    }
  #endif

  // Describes how to run the CPU BLAS routine (for correctness/performance comparison)
  #ifdef CLBLAST_REF_CBLAS
    static StatusCode RunReference2(const Arguments<T> &args, BuffersHost<T> &buffers_host, Queue &) {
      cblasXtrmm(convertToCBLAS(args.layout),
                 convertToCBLAS(args.side),
                 convertToCBLAS(args.triangle),
                 convertToCBLAS(args.a_transpose),
                 convertToCBLAS(args.diagonal),
                 args.m, args.n, args.alpha,
                 buffers_host.a_mat, args.a_offset, args.a_ld,
                 buffers_host.b_mat, args.b_offset, args.b_ld);
      return StatusCode::kSuccess;
    }
  #endif

  // Describes how to run the cuBLAS routine (for correctness/performance comparison)
  #ifdef CLBLAST_REF_CUBLAS
    static StatusCode RunReference3(const Arguments<T> &args, BuffersCUDA<T> &buffers, Queue &) {
      auto status = cublasXtrmm(reinterpret_cast<cublasHandle_t>(args.cublas_handle), args.layout,
                                convertToCUBLAS(args.side),
                                convertToCUBLAS(args.triangle),
                                convertToCUBLAS(args.a_transpose),
                                convertToCUBLAS(args.diagonal),
                                args.m, args.n, args.alpha,
                                buffers.a_mat, args.a_offset, args.a_ld,
                                buffers.b_mat, args.b_offset, args.b_ld);
      if (status == CUBLAS_STATUS_SUCCESS) { return StatusCode::kSuccess; } else { return StatusCode::kUnknownError; }
    }
  #endif

  // Describes how to download the results of the computation (more importantly: which buffer)
  static std::vector<T> DownloadResult(const Arguments<T> &args, Buffers<T> &buffers, Queue &queue) {
    std::vector<T> result(args.b_size, static_cast<T>(0));
    buffers.b_mat.Read(queue, args.b_size, result);
    return result;
  }

  // Describes how to compute the indices of the result buffer
  static size_t ResultID1(const Arguments<T> &args) { return args.m; }
  static size_t ResultID2(const Arguments<T> &args) { return args.n; }
  static size_t GetResultIndex(const Arguments<T> &args, const size_t id1, const size_t id2) {
    return (args.layout == Layout::kRowMajor) ?
           id1*args.b_ld + id2 + args.b_offset:
           id2*args.b_ld + id1 + args.b_offset;
  }

  // Describes how to compute performance metrics
  static size_t GetFlops(const Arguments<T> &args) {
    auto k = (args.side == Side::kLeft) ? args.m : args.n;
    return args.m * args.n * k;
  }
  static size_t GetBytes(const Arguments<T> &args) {
    auto k = (args.side == Side::kLeft) ? args.m : args.n;
    return (k*k + 2*args.m*args.n) * sizeof(T);
  }
};

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_ROUTINES_XTRMM_H_
#endif
