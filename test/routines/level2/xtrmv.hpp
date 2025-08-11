
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements a class with static methods to describe the Xtrmv routine. Examples of
// such 'descriptions' are how to calculate the size a of buffer or how to run the routine. These
// static methods are used by the correctness tester and the performance tester.
//
// =================================================================================================

#ifndef CLBLAST_TEST_ROUTINES_XTRMV_H_
#define CLBLAST_TEST_ROUTINES_XTRMV_H_

#include "test/routines/common.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class TestXtrmv {
 public:

  // The BLAS level: 1, 2, or 3
  static size_t BLASLevel() { return 2; }

  // The list of arguments relevant for this routine
  static std::vector<std::string> GetOptions() {
    return {kArgN,
            kArgLayout, kArgTriangle, kArgATransp, kArgDiagonal,
            kArgALeadDim, kArgXInc,
            kArgAOffset, kArgXOffset};
  }
  static std::vector<std::string> BuffersIn() { return {kBufMatA, kBufVecX}; }
  static std::vector<std::string> BuffersOut() { return {kBufVecX}; }

  // Describes how to obtain the sizes of the buffers
  static size_t GetSizeX(const Arguments<T> &args) {
    return args.n * args.x_inc + args.x_offset;
  }
  static size_t GetSizeA(const Arguments<T> &args) {
    return args.n * args.a_ld + args.a_offset;
  }

  // Describes how to set the sizes of all the buffers
  static void SetSizes(Arguments<T> &args, Queue&) {
    args.a_size = GetSizeA(args);
    args.x_size = GetSizeX(args);
  }

  // Describes what the default values of the leading dimensions of the matrices are
  static size_t DefaultLDA(const Arguments<T> &args) { return args.n; }
  static size_t DefaultLDB(const Arguments<T> &) { return 1; } // N/A for this routine
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
      auto status = Trmv<T>(args.layout, args.triangle, args.a_transpose, args.diagonal,
                            args.n,
                            buffers.a_mat(), args.a_offset, args.a_ld,
                            buffers.x_vec(), args.x_offset, args.x_inc,
                            &queue_plain, &event);
      if (status == StatusCode::kSuccess) { clWaitForEvents(1, &event); clReleaseEvent(event); }
    #elif CUDA_API
      auto status = Trmv<T>(args.layout, args.triangle, args.a_transpose, args.diagonal,
                            args.n,
                            buffers.a_mat(), args.a_offset, args.a_ld,
                            buffers.x_vec(), args.x_offset, args.x_inc,
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
      auto status = clblasXtrmv<T>(convertToCLBLAS(args.layout),
                                   convertToCLBLAS(args.triangle),
                                   convertToCLBLAS(args.a_transpose),
                                   convertToCLBLAS(args.diagonal),
                                   args.n,
                                   buffers.a_mat, args.a_offset, args.a_ld,
                                   buffers.x_vec, args.x_offset, args.x_inc,
                                   1, &queue_plain, 0, nullptr, &event);
      clWaitForEvents(1, &event);
      return static_cast<StatusCode>(status);
    }
  #endif

  // Describes how to run the CPU BLAS routine (for correctness/performance comparison)
  #ifdef CLBLAST_REF_CBLAS
    static StatusCode RunReference2(const Arguments<T> &args, BuffersHost<T> &buffers_host, Queue &) {
      cblasXtrmv(convertToCBLAS(args.layout),
                 convertToCBLAS(args.triangle),
                 convertToCBLAS(args.a_transpose),
                 convertToCBLAS(args.diagonal),
                 args.n,
                 buffers_host.a_mat, args.a_offset, args.a_ld,
                 buffers_host.x_vec, args.x_offset, args.x_inc);
      return StatusCode::kSuccess;
    }
  #endif

  // Describes how to run the cuBLAS routine (for correctness/performance comparison)
  #ifdef CLBLAST_REF_CUBLAS
    static StatusCode RunReference3(const Arguments<T> &args, BuffersCUDA<T> &buffers, Queue &) {
      auto status = cublasXtrmv(reinterpret_cast<cublasHandle_t>(args.cublas_handle), args.layout,
                                convertToCUBLAS(args.triangle),
                                convertToCUBLAS(args.a_transpose),
                                convertToCUBLAS(args.diagonal),
                                args.n,
                                buffers.a_mat, args.a_offset, args.a_ld,
                                buffers.x_vec, args.x_offset, args.x_inc);
      if (status == CUBLAS_STATUS_SUCCESS) { return StatusCode::kSuccess; } else { return StatusCode::kUnknownError; }
    }
  #endif

  // Describes how to download the results of the computation (more importantly: which buffer)
  static std::vector<T> DownloadResult(const Arguments<T> &args, Buffers<T> &buffers, Queue &queue) {
    std::vector<T> result(args.x_size, static_cast<T>(0));
    buffers.x_vec.Read(queue, args.x_size, result);
    return result;
  }

  // Describes how to compute the indices of the result buffer
  static size_t ResultID1(const Arguments<T> &args) {
    return args.n;
  }
  static size_t ResultID2(const Arguments<T> &) { return 1; } // N/A for this routine
  static size_t GetResultIndex(const Arguments<T> &args, const size_t id1, const size_t) {
    return id1*args.x_inc + args.x_offset;
  }

  // Describes how to compute performance metrics
  static size_t GetFlops(const Arguments<T> &args) {
    return 2 * args.n * args.n;
  }
  static size_t GetBytes(const Arguments<T> &args) {
    return (args.n*args.n + 2*args.n + args.n) * sizeof(T);
  }
};

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_ROUTINES_XTRMV_H_
#endif
